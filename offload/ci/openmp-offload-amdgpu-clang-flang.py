#! /usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

# Adapt to location in source tree
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
    with w.step("configure-openmp", halt_on_fail=True):
        w.run_command(
            [
                "cmake",
                f"-S{w.in_llvmsrc('llvm')}",
                f"-B{llvmbuilddir}",
                f"-C{w.in_llvmsrc('offload/cmake/caches/AMDGPUBot.cmake')}",
                "-GNinja",
                f"-DLLVM_LIT_ARGS=-vv --show-unsupported --show-xfail -j {w.jobs} --time-tests --timeout 100",
                f"-DCMAKE_INSTALL_PREFIX={w.in_workdir(llvminstalldir)}",
                "-DFLANG_RUNTIME_F128_MATH_LIB=libquadmath",
                "-DCMAKE_CXX_STANDARD=17",
                "-DLIBOMPTARGET_PLUGINS_TO_BUILD=amdgpu;host",
                "-DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES=compiler-rt;openmp",
                f"-DRUNTIMES_amdgcn-amd-amdhsa_CACHE_FILES={w.in_llvmsrc('compiler-rt')}/cmake/caches/GPU.cmake\;{w.in_llvmsrc('libcxx')}/cmake/caches/AMDGPU.cmake CACHE STRING '')",
            ]
        )

    with w.step("compile-openmp", halt_on_fail=True):
        w.run_ninja(builddir=llvmbuilddir, ccache_stats=True)

    with w.step("test-openmp"):
        w.run_ninja(
            ["check-openmp"], add_env={"HSA_ENABLE_SDMA": "0"}, builddir=llvmbuilddir
        )

    with w.step("Add check check-offload"):
        w.run_ninja(
            ["check-offload"], add_env={"HSA_ENABLE_SDMA": "0"}, builddir=llvmbuilddir
        )

    with w.step("LLVM: Install", halt_on_fail=True):
        w.run_ninja(["install"], builddir=llvmbuilddir)
