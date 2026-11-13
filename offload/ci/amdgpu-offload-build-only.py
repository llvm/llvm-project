#! /usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

# Used by a fast AMDGPU offload for build breakage signal only.

# Adapt to location in source tree
llvmsrcroot = os.path.normpath(f"{__file__}/../../..")

sys.path.insert(0, os.path.join(llvmsrcroot, ".ci/buildbot"))
import worker

llvmbuilddir = "llvm.build"

# Set to False to park this builder without changing its buildbot master
BUILDER_ENABLED = True

with worker.run(
    __file__,
    llvmsrcroot,
    cachefile="offload/cmake/caches/AMDGPUBot.cmake",
    clobberpaths=[llvmbuilddir],
    workerjobs=32,
    incremental=False,
) as w:
    if not BUILDER_ENABLED:
        with w.step("DISABLED: builder parked, see BUILDER_ENABLED"):
            w.report("This builder is intentionally parked; not building.")
    else:
        with w.step("configure llvm", halt_on_fail=True):
            w.run_command(
                [
                    "cmake",
                    "-GNinja",
                    f"-S{w.in_llvmsrc('llvm')}",
                    f"-B{llvmbuilddir}",
                    f"-C{w.cachefile}",
                ]
            )

        with w.step("build llvm", halt_on_fail=True):
            w.run_ninja(builddir=llvmbuilddir, ccache_stats=True)
