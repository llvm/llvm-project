# TestExclusivitySuppression.py
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
Test suppression of dynamic exclusivity enforcement
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2

import sys
if sys.version_info.major == 2:
    import commands as subprocess
else:
    import subprocess

def execute_command(command):
    (exit_status, output) = subprocess.getstatusoutput(command)
    return exit_status

class TestExclusivitySuppression(TestBase):

    # Test that we can evaluate w.s.i at Breakpoint 1 without triggering
    # a failure due to exclusivity
    @swiftTest
    def test_basic_exclusivity_suppression(self):
        """Test that exclusively owned values can still be accessed"""

        self.build()
        (target, process, thread, bp1) = lldbutil.run_to_source_breakpoint(self,
                'Breakpoint 1', self.main_source_spec)

        frame = thread.frames[0]
        self.assertTrue(frame, "Frame 0 is valid.")

        self.check_expression(frame, "w.s.i", "8", use_summary=False)

    # Test that we properly handle nested expression evaluations by:
    # (1) Breaking at breakpoint 1
    # (2) Running 'expr get()' (which will hit breakpoint 2)
    # (3) Evaluating i at breakpoint 2 (this is a nested evaluation)
    # (4) Continuing the evaluation of 'expr get()' to return to bp 1
    # (5) Evaluating w.s.i again to check that finishing the nested expression
    #     did not prematurely re-enable exclusivity checks.
    @swiftTest
    def test_exclusivity_suppression_for_concurrent_expressions(self):
        """Test that exclusivity suppression works with concurrent expressions"""
        self.build()
        (target, process, thread, bp1) = lldbutil.run_to_source_breakpoint(self,
                'Breakpoint 1', self.main_source_spec)

        # We hit Breakpoint 1, then evaluate 'get()' to hit breakpoint 2.
        bp2 = target.BreakpointCreateBySourceRegex('Breakpoint 2',
                                                   self.main_source_spec)
        self.assertTrue(bp2.GetNumLocations() > 0, VALID_BREAKPOINT)

        opts = lldb.SBExpressionOptions()
        opts.SetIgnoreBreakpoints(False)
        thread.frame[0].EvaluateExpression('get()', opts)

        # Evaluate w.s.i at breakpoint 2 to check that exclusivity checking
        # is suppressed inside the nested expression
        self.check_expression(thread.frames[0], "i", "8", use_summary=False)

        # Return to breakpoint 1 and evaluate w.s.i again to check that
        # exclusivity checking is still suppressed
        self.dbg.HandleCommand('thread ret -x')
        self.check_expression(thread.frame[0], "w.s.i", "8", use_summary=False)

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def check_expression(self, frame, expression, expected_result, use_summary=True):
        value = frame.EvaluateExpression(expression)
        self.assertTrue(value.IsValid(), expression + " returned a valid value")
        if self.TraceOn():
            print(value.GetSummary())
            print(value.GetValue())
        if use_summary:
            answer = value.GetSummary()
        else:
            answer = value.GetValue()
        report_str = "%s expected: %s got: %s" % (
            expression, expected_result, answer)
        self.assertTrue(answer == expected_result, report_str)
