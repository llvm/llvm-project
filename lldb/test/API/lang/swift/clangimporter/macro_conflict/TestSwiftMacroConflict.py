# TestSwiftMacroConflict.py
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

class TestSwiftMacroConflict(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        # To ensure we hit the rebuild problem remove the cache to avoid caching.
        mod_cache = self.getBuildArtifact("my-clang-modules-cache")
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)

        self.runCmd('settings set symbols.use-swift-dwarfimporter false')
        self.runCmd('settings set symbols.clang-modules-cache-path "%s"'
                    % mod_cache)
        self.build()

        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('Bar.swift'),
            extra_images=['Foo', 'Bar'])
        bar_value = self.frame().EvaluateExpression("bar")
        self.expect("fr var bar", "correct bar", substrs=["23"])

        foo_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('Foo.swift'))
        process.Continue()

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, foo_breakpoint)
        frame = threads[0].GetFrameAtIndex(0)
        foo_value = frame.EvaluateExpression("foo")
        # One is expected to fail because we use the same -D define as above.
        self.assertTrue(foo_value.GetError().Success() ^
                        bar_value.GetError().Success())

        self.assertTrue(os.path.isdir(mod_cache), "module cache exists")
        lldb.SBDebugger.MemoryPressureDetected()

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def test_with_dwarfimporter(self):
        """
        With DWARFImporter installed, both variables should be visible.
        """
        # To ensure we hit the rebuild problem remove the cache to avoid caching.
        mod_cache = self.getBuildArtifact("my-clang-modules-cache")
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)

        self.runCmd('settings set symbols.use-swift-dwarfimporter true')
        self.runCmd('settings set symbols.clang-modules-cache-path "%s"'
                    % mod_cache)
        self.build()

        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('Bar.swift'),
            extra_images=['Foo', 'Bar'])
        self.expect("v bar", substrs=["23"])
        foo_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('Foo.swift'))
        process.Continue()
        self.expect("v foo", substrs=["42"])
        self.assertTrue(os.path.isdir(mod_cache), "module cache exists")
        lldb.SBDebugger.MemoryPressureDetected()
