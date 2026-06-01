#! /usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import sys

# Adapt to location in source tree
llvmsrcroot = os.path.normpath(f"{__file__}/../../..")

sys.path.insert(0, os.path.join(llvmsrcroot, ".ci/buildbot"))
import worker


llvmbuilddir = "llvm.build"
testsuitebuilddir = "TS-build"


def absolute_path(path):
    return os.path.abspath(os.path.expanduser(path))


def in_test_suite(test_suite_src, path):
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.join(test_suite_src, path)


def check_test_suite_build_dir(test_suite_src, test_suite_build_dir):
    if test_suite_build_dir == os.path.abspath(os.sep):
        raise RuntimeError("refusing to use filesystem root as test-suite build dir")
    if test_suite_build_dir == test_suite_src:
        raise RuntimeError("test-suite build dir must not be the test-suite source dir")


parser = argparse.ArgumentParser(
    description="Build LLVM and run llvm-test-suite External HIP TPL tests."
)
parser.add_argument(
    "--test-suite-src",
    default="/opt/botworker/llvm/llvm-test-suite",
    help="Path to the llvm-test-suite checkout.",
)
parser.add_argument(
    "--test-suite-externals",
    default="/opt/botworker/llvm/External",
    help="Path to the llvm-test-suite External dependencies directory.",
)
parser.add_argument(
    "--test-suite-build-dir",
    default=testsuitebuilddir,
    help="Build directory for llvm-test-suite; relative paths are under test-suite-src.",
)
parser.add_argument(
    "--amdgpu-arch",
    default="gfx90a",
    help="AMDGPU architecture to use for HIP External tests.",
)

with worker.run(
    __file__,
    llvmsrcroot,
    parser=parser,
    cachefile="offload/cmake/caches/AMDGPUBot.cmake",
    clobberpaths=[llvmbuilddir],
    workerjobs=32,
) as w:
    test_suite_src = absolute_path(w.args.test_suite_src)
    test_suite_externals = absolute_path(w.args.test_suite_externals)
    test_suite_build_dir = absolute_path(
        in_test_suite(test_suite_src, w.args.test_suite_build_dir)
    )
    check_test_suite_build_dir(test_suite_src, test_suite_build_dir)
    compiler_bin_dir = w.in_workdir(os.path.join(llvmbuilddir, "bin"))
    clang = os.path.join(compiler_bin_dir, "clang")
    clangxx = os.path.join(compiler_bin_dir, "clang++")

    with w.step("configure llvm", halt_on_fail=True):
        w.run_command(
            [
                "cmake",
                "-GNinja",
                f"-S{w.in_llvmsrc('llvm')}",
                f"-B{llvmbuilddir}",
                f"-C{w.cachefile}",
                f"-DLLVM_ENABLE_RUNTIMES=compiler-rt",
            ]
        )

    with w.step("compile llvm", halt_on_fail=True):
        w.run_ninja(builddir=llvmbuilddir)

    with w.step("update llvm-test-suite", halt_on_fail=True):
        if not os.path.isdir(test_suite_src):
            raise RuntimeError(f"directory does not exist: {test_suite_src}")
        w.run_command(["git", "-C", test_suite_src, "reset", "--hard", "origin/main"])
        w.run_command(["git", "-C", test_suite_src, "pull"])

    with w.step("clean llvm-test-suite build", halt_on_fail=True):
        if os.path.exists(test_suite_build_dir):
            w.rmtree(test_suite_build_dir)

    with w.step("configure llvm-test-suite", halt_on_fail=True):
        w.run_command(
            [
                "cmake",
                "-GNinja",
                f"-B{test_suite_build_dir}",
                f"-S{test_suite_src}",
                f"-DTEST_SUITE_EXTERNALS_DIR={test_suite_externals}",
                f"-DAMDGPU_ARCHS={w.args.amdgpu_arch}",
                f"-DTEST_SUITE_SUBDIRS=External",
                f"-DEXTERNAL_HIP_TESTS_KOKKOS=ON",
                f"-DCMAKE_C_COMPILER={clang}",
                f"-DCMAKE_CXX_COMPILER={clangxx}",
            ]
        )

    with w.step("build kokkos and kokkos test suite", halt_on_fail=True):
        w.run_command(
            [
                "cmake",
                "--build",
                test_suite_build_dir,
                "--parallel",
                "--target",
                "build-kokkos",
            ]
        )

    with w.step("run kokkos test suite", halt_on_fail=True):
        w.run_command(
            ["cmake", "--build", test_suite_build_dir, "--target", "test-kokkos"]
        )
