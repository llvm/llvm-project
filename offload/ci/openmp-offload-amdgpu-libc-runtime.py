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
                f"-C{w.in_llvmsrc('offload/cmake/caches/AMDGPUBot.cmake')}",
                "-GNinja",
                f"-DLLVM_LIT_ARGS=-vv --show-unsupported --show-xfail -j {w.jobs} --time-tests --timeout 100",
                f"-DCMAKE_INSTALL_PREFIX={w.in_workdir(llvminstalldir)}",
                "-DLLVM_TARGETS_TO_BUILD=X86;AMDGPU",
                "-DLLVM_ENABLE_PROJECTS=clang;lld",
                "-DLLVM_ENABLE_RUNTIMES=compiler-rt;libunwind;libc;libcxx;libcxxabi;openmp;offload",
                "-DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES=libc",
                "-DCMAKE_CXX_STANDARD=17",
                "-DLLVM_ENABLE_LIBCXX=ON",
                "-DLLVM_ENABLE_ZLIB=ON",
                "-DLLVM_ENABLE_Z3_SOLVER=OFF",
                "-DCLANG_DEFAULT_UNWINDLIB=libgcc",
                "-DLIBOMPTARGET_PLUGINS_TO_BUILD=amdgpu;host",
                "-DLIBCXX_ENABLE_SHARED=OFF",
                "-DLIBCXX_ENABLE_STATIC=ON",
                "-DLIBCXX_INSTALL_LIBRARY=OFF",
                "-DLIBCXX_INSTALL_HEADERS=OFF",
                "-DLIBCXXABI_ENABLE_SHARED=OFF",
                "-DLIBCXXABI_ENABLE_STATIC=ON",
                "-DLIBCXXABI_INSTALL_STATIC_LIBRARY=OFF",
                "-DRUNTIMES_amdgcn-amd-amdhsa_LIBC_GPU_TEST_JOBS=4",
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

    with w.step("run check-libc-amdgcn-amd-amdhsa"):
        w.run_ninja(
            ["check-libc-amdgcn-amd-amdhsa"],
            add_env={"HSA_ENABLE_SDMA": "0"},
            builddir=llvmbuilddir,
        )

    with w.step("LLVM: Install", halt_on_fail=True):
        w.run_ninja(["install"], builddir=llvmbuilddir)
