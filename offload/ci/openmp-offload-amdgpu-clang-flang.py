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
                "-GNinja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DLLVM_ENABLE_ASSERTIONS=ON",
                f"-DLLVM_LIT_ARGS=-vv --show-unsupported --show-xfail -j {w.jobs} --time-tests --timeout 100",
                f"-DCMAKE_INSTALL_PREFIX={w.in_workdir(llvminstalldir)}",
                "-DCLANG_DEFAULT_LINKER=lld",
                "-DLLVM_TARGETS_TO_BUILD=X86;AMDGPU",
                "-DLLVM_ENABLE_ASSERTIONS=ON",
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DFLANG_RUNTIME_F128_MATH_LIB=libquadmath",
                "-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON",
                "-DCMAKE_CXX_STANDARD=17",
                "-DBUILD_SHARED_LIBS=ON",
                "-DLIBOMPTARGET_PLUGINS_TO_BUILD=amdgpu;host",
                "-DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES=compiler-rt;openmp",
                "-DLLVM_RUNTIME_TARGETS=default;amdgcn-amd-amdhsa",
                "-DCOMPILER_RT_BUILD_ORC=OFF",
                "-DCOMPILER_RT_BUILD_XRAY=OFF",
                "-DCOMPILER_RT_BUILD_MEMPROF=OFF",
                "-DCOMPILER_RT_BUILD_LIBFUZZER=OFF",
                "-DCOMPILER_RT_BUILD_SANITIZERS=ON",
                "-DLLVM_ENABLE_PROJECTS=clang;lld;mlir;flang;llvm",
                "-DLLVM_ENABLE_RUNTIMES=flang-rt;offload;compiler-rt;openmp",
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
