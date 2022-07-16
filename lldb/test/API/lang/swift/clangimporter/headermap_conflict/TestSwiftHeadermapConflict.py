# TestSwiftHeadermapConflict.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2018 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2
import shutil

class TestSwiftHeadermapConflict(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @skipIf(bugnumber="rdar://60396797",
            setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        # To ensure we hit the rebuild problem remove the cache to avoid caching.
        mod_cache = self.getBuildArtifact("my-clang-modules-cache")
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)

        self.runCmd("settings set symbols.use-swift-dwarfimporter false")
        self.runCmd('settings set symbols.clang-modules-cache-path "%s"'
                    % mod_cache)
        self.build()

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'),
            extra_images=['dylib'])
        b_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('dylib.swift'))


        # This is expected to succeed because ClangImporter was set up
        # with the flags from the main executable.
        self.expect("expr foo", "expected result", substrs=["42"])
        self.assertTrue(os.path.isdir(mod_cache), "module cache exists")

        process.Continue()
        threads = lldbutil.get_threads_stopped_at_breakpoint(process,
                                                             b_breakpoint)
        frame = threads[0].GetFrameAtIndex(0)
        value = frame.EvaluateExpression("foo")
        # This is expected to fail because ClangImporter is *still*
        # set up with the flags from the main executable.
        self.assertTrue((not value.GetError().Success()) or
                         not value.GetSummary())
        self.runCmd("settings set symbols.use-swift-dwarfimporter true")
