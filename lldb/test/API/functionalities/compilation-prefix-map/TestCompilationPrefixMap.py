# TestCompilationPrefixMap.py
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""
Test that LLDB auto-loads compilation-prefix-map.json to resolve remapped
source paths without requiring manual `settings set target.source-map`.
"""
import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestCompilationPrefixMap(TestBase):
    @skipIfWindows
    @skipIfHostIncompatibleWithTarget
    def test_compilation_prefix_map(self):
        """
        Build a binary with -fdebug-prefix-map remapping the source directory
        to /fake/srcdir, place compilation-prefix-map.json next to the binary
        mapping /fake/srcdir back to the real source directory, and verify that
        LLDB resolves a source-line breakpoint without any manual source-map
        configuration.
        """
        self.build()

        src_dir = self.getSourceDir()

        log = self.getBuildArtifact("module.log")
        self.runCmd('log enable lldb module -f "%s"' % log)

        source_spec = lldb.SBFileSpec(os.path.join(src_dir, "main.c"))
        lldbutil.run_to_source_breakpoint(self, "return x", source_spec)

        self.filecheck_log(log, __file__)


#       CHECK: found compilation-prefix-map.json
#       CHECK: applying prefix map: '/fake/srcdir'
