#! /usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

llvmsrcroot = os.path.normpath(f"{__file__}/../../..")
sys.path.insert(0, os.path.join(llvmsrcroot, ".ci/buildbot"))
import worker

llvmbuilddir = "llvm.build"
llvminstalldir = "llvm.inst"

with worker.run(
    __file__,
    llvmsrcroot,
    clobberpaths=[llvmbuilddir, llvminstalldir],
    workerjobs=64,
) as w:
    with w.step("configure llvm", halt_on_fail=True):
        w.run_command(
            [
                "cmake",
                f"-S{w.in_llvmsrc('llvm')}",
                f"-B{llvmbuilddir}",
                f"-C{w.in_llvmsrc('offload/cmake/caches/AMDGPULibcBot.cmake')}",
                "-GNinja",
                f"-DLLVM_LIT_ARGS=-vv --show-unsupported --show-xfail -j {w.jobs} --time-tests --timeout 100",
                f"-DCMAKE_INSTALL_PREFIX={w.in_workdir(llvminstalldir)}",
                f"-DRUNTIMES_amdgpu-amd-amdhsa_CMAKE_CROSSCOMPILING_EMULATOR={w.in_workdir(llvmbuilddir)}/bin/llvm-gpu-loader",
            ]
        )

    with w.step("compile llvm", halt_on_fail=True):
        w.run_ninja(builddir=llvmbuilddir, ccache_stats=True)

    with w.step("run check-openmp"):
        w.run_ninja(
            ["check-openmp"], add_env={"HSA_ENABLE_SDMA": "0"}, builddir=llvmbuilddir
        )

    with w.step("run check-clang"):
        w.run_ninja(
            ["check-clang"], add_env={"HSA_ENABLE_SDMA": "0"}, builddir=llvmbuilddir
        )

    with w.step("run check-llvm"):
        w.run_ninja(
            ["check-llvm"], add_env={"HSA_ENABLE_SDMA": "0"}, builddir=llvmbuilddir
        )

    with w.step("run check-lld"):
        w.run_ninja(
            ["check-lld"], add_env={"HSA_ENABLE_SDMA": "0"}, builddir=llvmbuilddir
        )

    with w.step("run check-libc-amdgpu-amd-amdhsa"):
        w.run_ninja(
            ["check-libc-amdgpu-amd-amdhsa"],
            add_env={"HSA_ENABLE_SDMA": "0"},
            builddir=llvmbuilddir,
        )

    with w.step("run check-compiler-rt-amdgpu-amd-amdhsa"):
        w.run_ninja(
            ["check-compiler-rt-amdgpu-amd-amdhsa"],
            add_env={"HSA_ENABLE_SDMA": "0"},
            builddir=llvmbuilddir,
        )

    with w.step("LLVM: Install", halt_on_fail=True):
        w.run_ninja(["install"], builddir=llvmbuilddir)
