# TestSwiftGenericExpressions.py
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
Test expressions in generic contexts
"""
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftGenericExpressions(lldbtest.TestBase):

    def setUp(self):
        lldbtest.TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    @swiftTest
    def test_generic_expressions(self):
        """Test expressions in generic contexts"""
        self.build()
        self.do_test()

    @swiftTest
    def test_ivars_in_generic_expressions(self):
        """Test ivar access through expressions in generic contexts"""
        self.build()
        self.do_ivar_test()

    def check_expression(self, expression, expected_result, use_summary=True):
        opts = lldb.SBExpressionOptions()
        opts.SetFetchDynamicValue(lldb.eDynamicCanRunTarget)
        value = self.frame().EvaluateExpression(expression, opts)
        self.assertTrue(value.IsValid(), expression + "returned a valid value")

        self.assertSuccess(value.GetError(), "expression failed")
        if self.TraceOn():
            print(value.GetSummary())
            print(value.GetValue())

        if use_summary:
            answer = value.GetSummary()
        else:
            answer = value.GetValue()
        report_str = "Use summary: %d %s expected: %s got: %s" % (
            use_summary, expression, expected_result, answer)
        self.assertTrue(answer == expected_result, report_str)

    def do_test(self):
        """Test expressions in generic contexts"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        breakpoints = [None]

        # Set the breakpoints
        for i in range(1, 7):
            breakpoints.append(target.BreakpointCreateBySourceRegex(
                "breakpoint " + str(i), self.main_source_spec))
            self.assertTrue(
                breakpoints[i].GetNumLocations() > 0,
                lldbtest.VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, lldbtest.PROCESS_IS_VALID)

        # Check each context
        for i in range(1, 7):
            # Frame #0 should be at our breakpoint.
            threads = lldbutil.get_threads_stopped_at_breakpoint(
                process, breakpoints[i])

            self.assertTrue(len(threads) == 1)
            self.check_expression("i", str(i), use_summary=False)

            self.runCmd("continue")

    def do_ivar_test(self):
        """Test expressions in generic contexts"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        breakpoints = []

        # Set the breakpoints only in the class functions:
        class_bkpts = [2, 3, 5, 6]
        for i in range(0, 4):
            breakpoints.append(target.BreakpointCreateBySourceRegex(
                "breakpoint " + str(class_bkpts[i]), self.main_source_spec))
            self.assertTrue(
                breakpoints[i].GetNumLocations() > 0,
                lldbtest.VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, lldbtest.PROCESS_IS_VALID)

        # Check each context
        for i in range(0, 4):
            # Frame #0 should be at our breakpoint.
            threads = lldbutil.get_threads_stopped_at_breakpoint(
                process, breakpoints[i])

            self.assertTrue(len(threads) == 1)
            self.check_expression(
                "m_t", str(class_bkpts[i]), use_summary=False)
            self.check_expression(
                "m_s.m_s", str(class_bkpts[i]), use_summary=False)

            self.runCmd("continue")

