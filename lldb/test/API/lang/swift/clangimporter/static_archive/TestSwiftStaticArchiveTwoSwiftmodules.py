# TestSwiftStaticArchiveTwoSwiftmodules.py
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

class TestSwiftStaticArchiveTwoSwiftmodules(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints.
        foo_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('Foo.swift'))
        bar_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('Bar.swift'))
        self.assertTrue(foo_breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)
        self.assertTrue(bar_breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch.
        process = target.LaunchSimple(None, None, os.getcwd())

        # This test tests that the search paths from all swiftmodules
        # that are part of the main binary are honored.
        self.expect("fr var foo", "expected result", substrs=["23"])
        self.expect("p foo", "expected result", substrs=["$R0", "i", "23"])
        process.Continue()
        self.expect("fr var bar", "expected result", substrs=["42"])
        self.expect("p bar", "expected result", substrs=["j", "42"])
