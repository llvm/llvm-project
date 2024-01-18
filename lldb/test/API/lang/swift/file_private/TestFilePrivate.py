# TestFilePrivate.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Test that we find the right file-local private decls using the discriminator
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os


class TestFilePrivate(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.a_source = "a.swift"
        self.a_source_spec = lldb.SBFileSpec(self.a_source)
        self.b_source = "b.swift"
        self.b_source_spec = lldb.SBFileSpec(self.b_source)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    @swiftTest
    def test(self):
        """Test that we find the right file-local private decls using the discriminator"""
        self.build()

        # Create the target
        target = self.dbg.CreateTarget(self.getBuildArtifact())
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        a_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', self.a_source_spec)
        self.assertTrue(a_breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)
        b_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', self.b_source_spec)
        self.assertTrue(b_breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)
        main_breakpoint = target.BreakpointCreateBySourceRegex(
            'break here', self.main_source_spec)
        self.assertTrue(
            main_breakpoint.GetNumLocations() > 0,
            VALID_BREAKPOINT)

        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, a_breakpoint)

        self.assertTrue(len(threads) == 1)
        lldbutil.check_expression(self, self.frame(), "privateVariable", "\"five\"", use_summary=True)

        process.Continue()
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, b_breakpoint)

        self.assertTrue(len(threads) == 1)
        lldbutil.check_expression(self, self.frame(), "privateVariable", "3", use_summary=False)

        process.Continue()
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, main_breakpoint)

        self.assertTrue(len(threads) == 1)

        lldbutil.check_expression(self, self.frame(), "privateVariable", None, use_summary=True)
        lldbutil.check_expression(self, self.frame(), "privateVariable as Int", "3", use_summary=False)
        lldbutil.check_expression(self, self.frame(), "privateVariable as String", "\"five\"", use_summary=True)
