#! /usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Check Polly optimizations on llvm-test-suite"""

import os
import sys

# Adapt to location in source tree
llvmsrcroot = os.path.normpath(f"{__file__}/../../..")

sys.path.insert(0, os.path.join(llvmsrcroot, ".ci/buildbot"))
import worker

llvmbuilddir = "llvm.build"
llvminstalldir = "llvm.install"
testsuitesrcdir = "testsuite.src"
testsuitebuilddir = "testsuite.build"

with worker.run(
    __file__,
    llvmsrcroot,
    cachefile="polly/ci/polly-x86_64-linux-test-suite.cmake",
    clobberpaths=[llvmbuilddir, testsuitebuilddir, llvminstalldir],
    incremental=True,
) as w:
    with w.step("configure-llvm", halt_on_fail=True):
        cmakecmd = [
            "cmake",
            f"-S{w.in_llvmsrc('llvm')}",
            f"-B{llvmbuilddir}",
            "-GNinja",
            f"-C{w.in_llvmsrc(w.cachefile)}",
            f"-DCMAKE_INSTALL_PREFIX={llvminstalldir}",
        ]
        if w.jobs:
            cmakecmd.append(f"-DLLVM_LIT_ARGS=-sv;-j{w.jobs}")
        w.run_command(cmakecmd)

    with w.step("build-llvm", halt_on_fail=True):
        w.run_ninja(builddir=llvmbuilddir, ccache_stats=True)

    with w.step("check-polly"):
        w.run_ninja(["check-polly"], builddir=llvmbuilddir)

    with w.step("install-llvm", halt_on_fail=True):
        w.run_ninja(["install"], builddir=llvmbuilddir)

    with w.step("checkout-testsuite", halt_on_fail=True):
        w.checkout("https://github.com/llvm/llvm-test-suite", testsuitesrcdir)

    with w.step("configure-testsuite", halt_on_fail=True):
        jobsarg = f";-j{w.jobs}" if w.jobs else ""
        w.run_command(
            [
                "cmake",
                f"-S{testsuitesrcdir}",
                f"-B{testsuitebuilddir}",
                "-GNinja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_C_COMPILER={os.path.abspath(llvminstalldir)}/bin/clang",
                f"-DCMAKE_CXX_COMPILER={os.path.abspath(llvminstalldir)}/bin/clang++",
                f"-DTEST_SUITE_LIT={os.path.abspath(llvmbuilddir)}/bin/llvm-lit",
                f"-DTEST_SUITE_LLVM_SIZE={os.path.abspath(llvmbuilddir)}/bin/llvm-size",
                "-DTEST_SUITE_EXTRA_C_FLAGS=-Wno-unused-command-line-argument -mllvm -polly",
                "-DTEST_SUITE_EXTRA_CXX_FLAGS=-Wno-unused-command-line-argument -mllvm -polly",
                f"-DLLVM_LIT_ARGS=-sv{jobsarg};-o;report.json",
            ]
        )

    with w.step("build-testsuite", halt_on_fail=True):
        w.run_ninja(builddir=testsuitebuilddir)

    with w.step("check-testsuite"):
        w.run_ninja(["check"], builddir=testsuitebuilddir)
