#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 OpenTuner 自动调优 CPython 的 GCC 编译 / 链接选项,以 pyperformance 作为评测目标。

原理
----
CPython 的 Makefile 里最终编译命令为::

    $(CC) ... $(BASECFLAGS) $(OPT) $(CONFIGURE_CFLAGS) $(CFLAGS) $(EXTRA_CFLAGS) ...
    $(CC) ... $(CONFIGURE_CFLAGS_NODIST) $(CFLAGS_NODIST) ...   # NODIST 部分在最后

其中:
* ``OPT``              由 ``--enable-optimizations`` 决定,实际是 ``-DNDEBUG -g -fwrapv -O3 -Wall``;
* ``CONFIGURE_CFLAGS`` 就是你 configure 时通过环境变量 ``CFLAGS`` 传入的值;
* ``CONFIGURE_LDFLAGS`` 同理来自环境变量 ``LDFLAGS``;
* ``CONFIGURE_CFLAGS_NODIST`` / ``CONFIGURE_LDFLAGS_NODIST`` 里是 configure 自己追加的
  ``-fno-semantic-interposition -flto -fuse-linker-plugin -ffat-lto-objects
  -flto-partition=none`` 等(由 ``--with-lto`` 决定)。

由于 ``CONFIGURE_CFLAGS`` 位于 ``OPT``(-O3)*之后*,所以在 ``CFLAGS`` 里放 ``-O2`` 会
真正覆盖掉 ``-O3``;而我们想调优的 ``-f`` / ``--param`` / ``-march`` 选项也都放在
``CFLAGS`` 里,位置同样在 NODIST 之前、不会和 LTO/PGO 冲突。

注意:``-fno-semantic-interposition`` 和 ``-flto-partition=none`` 已经被
``--with-lto`` / ``--enable-optimizations`` 固定追加到 NODIST 里,并且位于用户
CFLAGS 之后,因此无法通过 CFLAGS 覆盖,本脚本也就不把它们列为可调参数。

每个配置会:
  1. 在独立的 build 目录里 configure + make + make install 出一个 python3.10;
  2. 用宿主机上的 pyperformance 对刚编译出的解释器跑基准;
  3. 读取 pyperformance 输出的 JSON,取所有 benchmark 中位数的几何平均作为分数
     (越小越快),返回给 OpenTuner。

用法示例
--------
    python3 cpython_pyperformance.py \
        --source-dir ~/gerrit/cpython \
        --build-root ./builds \
        --pyperformance-python python3 \
        --benchmarks 2to3,chameleon,django_template,json_loads,regex_compile \
        --fast \
        --jobs 16 \
        --parallelism 1 \
        --test-limit 200

说明:
* ``--fast``            让 pyperformance 用快速模式(每项只测一个值),大幅缩短单次采样;
* ``--benchmarks``      只跑一小批有代表性的 benchmark,进一步缩短采样时间;
* ``--test-limit N``    调优到第 N 个采样后停止(本 fork 里是“采样个数”,不是超时);
                        也可用 ``--stop-after SECONDS`` 按墙钟时间停止;
* ``--parallelism 1``   一次只 build 一个配置,避免并行 make 互相抢占 CPU;
* 单个配置的 build / 跑分超时由 ``--build-timeout`` / ``--run-timeout`` 控制
  (本 fork 的 ``--test-limit`` 已不再是单次超时,见上);
* 中断后重新运行会自动续跑(结果保存在 opentuner.db 里)。
* 每个配置的依赖会装到各自 venv 里,但 pip 会命中本地缓存,不会重复联网下载;
  跑完默认删除 venv 以节省磁盘,可用 ``--keep-venv`` 保留。
"""

from __future__ import print_function

import json
import logging
import math
import os
import shlex
import shutil

import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result

log = logging.getLogger("cpython_pyperformance")


# ---------------------------------------------------------------------------
# 可调优的 GCC -f 编译选项。
# 每个选项取值 on / off / default:
#   on       -> -f<flag>
#   off      -> -fno-<flag>
#   default  -> 不显式指定(交给 -O3 的默认行为)
# 这里刻意只挑了对 CPython 性能可能有影响、且 -fno- 形式都合法的选项,避免构建失败。
# ---------------------------------------------------------------------------
COMPILE_FLAGS = [
    "funroll-loops",
    "fipa-pta",
    "ftree-vectorize",
    "fomit-frame-pointer",
    "fpredictive-commoning",
    "fgcse-after-reload",
    "freorder-blocks-and-partition",
    "fdevirtualize-speculatively",
    "fgraphite-identity",
    "ftracer",
    "ftree-loop-distribute-patterns",
    "finline-functions",
]

# (名字, 最小值, 最大值) 的 GCC --param 参数。
GCC_PARAMS = [
    ("early-inlining-insns", 0, 300),
    ("inline-unit-growth", 0, 100),
    ("max-inline-insns-auto", 0, 100),
    ("predictable-branch-outcome", 0, 100),
]


class CpythonTuner(MeasurementInterface):
    def __init__(self, *pargs, **kwargs):
        kwargs.setdefault("program_name", "cpython")
        super(CpythonTuner, self).__init__(*pargs, **kwargs)
        # 以 (cflags, ldflags) 为 key 的构建缓存,避免同一配置重复 build。
        self._build_cache = {}

    # ------------------------------------------------------------------
    # 搜索空间
    # ------------------------------------------------------------------
    def manipulator(self):
        manipulator = ConfigurationManipulator()

        # 优化等级。放在 CFLAGS 里,位置在 OPT(-O3) 之后,可以真正覆盖它。
        manipulator.add_parameter(EnumParameter("opt", ["-O2", "-O3"]))
        # 是否针对本机 CPU 微架构(-march=native / -mtune=native)。
        manipulator.add_parameter(EnumParameter("march", ["default", "native"]))

        for flag in COMPILE_FLAGS:
            manipulator.add_parameter(
                EnumParameter(flag, ["on", "off", "default"]))

        for param, lo, hi in GCC_PARAMS:
            manipulator.add_parameter(IntegerParameter(param, lo, hi))

        return manipulator

    # ------------------------------------------------------------------
    # 把配置转成 CFLAGS / LDFLAGS 字符串
    # ------------------------------------------------------------------
    def _cfg_to_cflags(self, cfg):
        flags = [cfg["opt"]]
        if cfg["march"] == "native":
            flags += ["-march=native", "-mtune=native"]

        for flag in COMPILE_FLAGS:
            value = cfg[flag]
            if value == "on":
                flags.append("-f" + flag)
            elif value == "off":
                flags.append("-fno-" + flag)

        for param, _lo, _hi in GCC_PARAMS:
            flags.append("--param=%s=%d" % (param, cfg[param]))

        return " ".join(flags)

    def _cfg_to_ldflags(self, cfg):
        # 链接阶段目前没有额外可调项(如需,可在此扩展,例如 -Wl,-O1 等)。
        return ""

    # ------------------------------------------------------------------
    # 构建 CPython
    # ------------------------------------------------------------------
    def _build(self, build_dir, cflags, ldflags):
        prefix = os.path.join(build_dir, "install")
        configure = os.path.join(self.args.source_dir, "configure")
        logfile = os.path.join(build_dir, "build.log")

        os.makedirs(build_dir, exist_ok=True)

        cmd = (
            "set -e; "
            "cd {build_dir}; "
            "export CFLAGS={cflags}; "
            "export LDFLAGS={ldflags}; "
            "{configure} --prefix={prefix} {configure_flags}; "
            "make -j{jobs}; "
            "make install"
        ).format(
            build_dir=shlex.quote(build_dir),
            cflags=shlex.quote(cflags),
            ldflags=shlex.quote(ldflags),
            configure=shlex.quote(configure),
            prefix=shlex.quote(prefix),
            configure_flags=self.args.configure_flags,
            jobs=self.args.jobs,
        )
        cmd += " > {log} 2>&1".format(log=shlex.quote(logfile))

        log.info("building: CFLAGS=%s LDFLAGS=%s", cflags, ldflags)
        result = self.call_program(cmd, limit=self.args.build_timeout)

        if result["timeout"]:
            log.error("build 超时: %s", self._tail(logfile))
            return None
        if result["returncode"] != 0:
            log.error("build 失败 (returncode=%s):\n%s",
                      result["returncode"], self._tail(logfile))
            return None
        return prefix

    def _find_python(self, prefix):
        """在 install/bin 里找到编译出的 python3 可执行文件。"""
        bin_dir = os.path.join(prefix, "bin")
        if not os.path.isdir(bin_dir):
            return None
        candidates = sorted(
            f for f in os.listdir(bin_dir)
            if f.startswith("python3.") and "config" not in f
        )
        if candidates:
            return os.path.join(bin_dir, candidates[0])
        for name in ("python3", "python"):
            path = os.path.join(bin_dir, name)
            if os.path.exists(path):
                return path
        return None

    # ------------------------------------------------------------------
    # 运行 pyperformance 并解析分数
    # ------------------------------------------------------------------
    def _run_pyperformance(self, build_dir, python_bin, result_path):
        """运行 pyperformance,返回 "OK" / "TIMEOUT" / "ERROR"。"""
        logfile = os.path.join(build_dir, "pyperformance.log")

        cmd = [
            self.args.pyperformance_python, "-m", "pyperformance", "run",
            "--python", python_bin,
            "-o", result_path,
        ]
        if self.args.fast:
            cmd.append("-f")
        if self.args.benchmarks:
            cmd += ["-b", self.args.benchmarks]

        # 在 build_dir 里运行,让 pyperformance 的 venv/ 落到 build_dir 下,
        # 之后可以随 build_dir 一起清理。
        result = self.call_program(
            cmd, limit=self.args.run_timeout, cwd=build_dir)

        # 把 stdout/stderr 落盘,方便排查单个配置的跑分情况。
        with open(logfile, "wb") as fd:
            fd.write(result.get("stdout") or b"")
            fd.write(b"\n")
            fd.write(result.get("stderr") or b"")

        if result["timeout"]:
            log.error("pyperformance 超时: %s", self._tail(logfile))
            return "TIMEOUT"
        if result["returncode"] != 0:
            log.error("pyperformance 失败 (returncode=%s):\n%s",
                      result["returncode"], self._tail(logfile))
            return "ERROR"
        return "OK"

    def _parse_score(self, result_path):
        """返回所有 benchmark 中位数的几何平均(秒),越小越快。"""
        try:
            import pyperf
        except ImportError:
            log.error("宿主机 python 缺少 pyperf,无法解析结果")
            return None

        try:
            suite = pyperf.BenchmarkSuite.load(result_path)
        except Exception as exc:  # noqa: BLE001
            log.error("无法解析 pyperformance 结果 %s: %s", result_path, exc)
            return None

        times = []
        for bench in suite.get_benchmarks():
            median = bench.median()
            if median and median > 0:
                times.append(median)
        if not times:
            log.error("pyperformance 结果里没有有效的 benchmark 数据")
            return None

        return math.exp(sum(math.log(t) for t in times) / len(times))

    # ------------------------------------------------------------------
    # OpenTuner 入口:build + run
    # ------------------------------------------------------------------
    def run(self, desired_result, input, limit):  # noqa: A002
        cfg = desired_result.configuration.data
        cflags = self._cfg_to_cflags(cfg)
        ldflags = self._cfg_to_ldflags(cfg)

        cache_key = (cflags, ldflags)
        cached = self._build_cache.get(cache_key)
        if cached is not None:
            build_dir, python_bin = cached
        else:
            build_dir = os.path.join(
                self.args.build_root, "build-%d" % desired_result.id)
            prefix = self._build(build_dir, cflags, ldflags)
            if prefix is None:
                return Result(state="ERROR", time=float("inf"))

            python_bin = self._find_python(prefix)
            if python_bin is None:
                log.error("install 目录里找不到 python3 可执行文件")
                return Result(state="ERROR", time=float("inf"))
            self._build_cache[cache_key] = (build_dir, python_bin)

        result_path = os.path.join(build_dir, "pyperformance.json")
        status = self._run_pyperformance(build_dir, python_bin, result_path)

        # 跑完清理 venv,节省磁盘(依赖已进 pip 缓存,重测同配置会重新安装)。
        venv_dir = os.path.join(build_dir, "venv")
        if not self.args.keep_venv and os.path.isdir(venv_dir):
            shutil.rmtree(venv_dir, ignore_errors=True)

        if status != "OK":
            return Result(state=status, time=float("inf"))

        score = self._parse_score(result_path)
        if score is None:
            return Result(state="ERROR", time=float("inf"))

        log.info("score (geomean of medians) = %.4f s", score)
        return Result(time=score)

    # ------------------------------------------------------------------
    # 结束时输出最优配置
    # ------------------------------------------------------------------
    def save_final_config(self, configuration):
        cfg = configuration.data
        cflags = self._cfg_to_cflags(cfg)
        ldflags = self._cfg_to_ldflags(cfg)

        os.makedirs(self.args.output_dir, exist_ok=True)
        out_json = os.path.join(self.args.output_dir, "best_config.json")
        out_sh = os.path.join(self.args.output_dir, "best_flags.sh")

        with open(out_json, "w") as fd:
            json.dump(cfg, fd, indent=2, sort_keys=True)

        with open(out_sh, "w") as fd:
            fd.write("#!/bin/sh\n")
            fd.write("# 由 OpenTuner 自动调优得到的最优 CPython 编译/链接选项\n")
            fd.write("export CFLAGS=%s\n" % shlex.quote(cflags))
            fd.write("export LDFLAGS=%s\n" % shlex.quote(ldflags))
            fd.write(
                "cd build/best && %s/configure --prefix=$PWD/install %s && "
                "make -j && make install\n"
                % (shlex.quote(self.args.source_dir),
                   self.args.configure_flags))

        print("最优配置已写入:")
        print("  ", out_json)
        print("  ", out_sh)
        print("  CFLAGS =", cflags)
        print("  LDFLAGS =", ldflags)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _tail(path, n=40):
        try:
            with open(path, "rb") as fd:
                lines = fd.readlines()
            return b"".join(lines[-n:]).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return ""


def _add_args(argparser):
    argparser.add_argument(
        "--source-dir", default=os.path.expanduser("~/gerrit/cpython"),
        help="CPython 源码目录(包含 configure)")
    argparser.add_argument(
        "--build-root", default="./builds",
        help="构建目录,每个配置一个 build-<id> 子目录")
    argparser.add_argument(
        "--pyperformance-python", default="python3",
        help="用来运行 pyperformance 的宿主机 python(已装好 pyperformance)")
    argparser.add_argument(
        "--configure-flags", default="--enable-optimizations --with-lto",
        help="固定的 configure 参数(PGO/LTO 等)")
    argparser.add_argument(
        "--jobs", type=int, default=os.cpu_count(),
        help="make -j 的并行度")
    argparser.add_argument(
        "--benchmarks", default=None,
        help="逗号分隔的 benchmark 列表,传给 pyperformance -b(默认全部)")
    argparser.add_argument(
        "--fast", action="store_true",
        help="pyperformance 快速模式,缩短单次采样")
    argparser.add_argument(
        "--build-timeout", type=float, default=3600.0,
        help="单次 build 的超时(秒)")
    argparser.add_argument(
        "--run-timeout", type=float, default=7200.0,
        help="单次 pyperformance 跑分的超时(秒)")
    argparser.add_argument(
        "--keep-venv", action="store_true",
        help="跑完后保留 pyperformance 生成的 venv(默认删除以节省磁盘)")
    argparser.add_argument(
        "--output-dir", default=".",
        help="最优配置输出目录")
    return argparser


if __name__ == "__main__":
    opentuner.init_logging()
    argparser = opentuner.default_argparser()
    argparser = _add_args(argparser)
    CpythonTuner.main(argparser.parse_args())
