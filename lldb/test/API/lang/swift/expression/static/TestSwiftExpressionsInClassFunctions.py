# TestSwiftExpressionsInClassFunctions.py
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
Test expressions in the context of class functions
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftExpressionsInClassFunctions(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    def check_expression(self, expression, expected_result, use_summary=True):
        opts = lldb.SBExpressionOptions()
        opts.SetFetchDynamicValue(lldb.eDynamicCanRunTarget)
        value = self.frame().EvaluateExpression(expression, opts)
        self.assertTrue(value.IsValid(), expression + "returned a valid value")
        if use_summary:
            answer = value.GetSummary()
        else:
            answer = value.GetValue()
        report_str = value.GetError()
        self.assertEquals(answer, expected_result, report_str)


    @swiftTest
    def test_expressions_in_class_functions(self):
        """Test expressions in class func contexts"""
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact())
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoints = [None]
        for i in range(1, 8):
            breakpoints.append(
                target.BreakpointCreateBySourceRegex(
                    "breakpoint " + str(i), lldb.SBFileSpec('main.swift')))
            self.assertTrue(
                breakpoints[i].GetNumLocations() > 0,
                "Didn't get valid breakpoint for %s" % (str(i)))

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Check each context
        for i in range(1, 8):
            # Frame #0 should be at our breakpoint.
            threads = lldbutil.get_threads_stopped_at_breakpoint(
                process, breakpoints[i])

            self.assertTrue(len(threads) == 1)
            self.check_expression("i", str(i), False)
            if i == 6:
              self.check_expression("self", "a.H<Int>")
              frame = threads[0].GetFrameAtIndex(0)
              lldbutil.check_variable(self, frame.FindVariable("self"),
                                      use_dynamic=True,
                                      # FIXME: This should be '@thick a.H<Swift.Int>.Type'
                                      # but valobj.GetDynamicValue(lldb.eDynamicCanRunTarget)
                                      # doesn't seem to do its job.
                                      # rdar://problem/69889462
                                      typename='@thin a.H<τ_0_0>.Type')

            self.runCmd("continue")

