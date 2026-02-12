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

with worker.run(
    __file__,
    llvmsrcroot,
    clobberpaths=[llvmbuilddir],
    incremental=True,
) as w:
    with w.step("configure-llvm", halt_on_fail=True):
        cmakecmd = [
            "cmake",
            f"-S{w.in_llvmsrc('llvm')}",
            f"-B{llvmbuilddir}",
            "-GNinja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DLLVM_ENABLE_PROJECTS=polly",
            "-DLLVM_TARGETS_TO_BUILD=X86",
            "-DLLVM_ENABLE_LLD=ON",
            "-DLLVM_ENABLE_ASSERTIONS=ON",
            "-DLLVM_POLLY_LINK_INTO_TOOLS=OFF",
            "-DBUILD_SHARED_LIBS=ON",
            "-DLLVM_BUILD_LLVM_DYLIB=OFF",
            "-DLLVM_LINK_LLVM_DYLIB=OFF",
        ]
        if w.jobs:
            cmakecmd.append(f"-DLLVM_LIT_ARGS=-sv;-j{w.jobs}")
        w.run_command(cmakecmd)

    with w.step("build-llvm", halt_on_fail=True):
        w.run_ninja(builddir=llvmbuilddir, ccache_stats=True)

    with w.step("check-polly"):
        w.run_ninja(["check-polly"], builddir=llvmbuilddir)
