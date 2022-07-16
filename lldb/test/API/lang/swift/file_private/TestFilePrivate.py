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

    def check_expression(self, expression, expected_result, use_summary=True):
        value = self.frame().EvaluateExpression(expression)
        self.assertTrue(value.IsValid(), expression + "returned a valid value")
        # print value.GetSummary()
        # print value.GetValue()
        if use_summary:
            answer = value.GetSummary()
        else:
            answer = value.GetValue()
        report_str = "%s expected: %s got: %s" % (
            expression, expected_result, answer)
        self.assertTrue(answer == expected_result, report_str)

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
        self.check_expression("privateVariable", "\"five\"")

        process.Continue()
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, b_breakpoint)

        self.assertTrue(len(threads) == 1)
        self.check_expression("privateVariable", "3", False)

        process.Continue()
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, main_breakpoint)

        self.assertTrue(len(threads) == 1)

        self.check_expression("privateVariable", None)
        self.check_expression("privateVariable as Int", "3", False)
        self.check_expression("privateVariable as String", "\"five\"")

