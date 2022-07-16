# TestDefiningOverloadedFunctions.py
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
Tests that we can define overloaded functions in the expression parser/REPL
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestDefiningOverloadedFunctions(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    def check_expression(self, expression, expected_result, use_summary=True):
        value = self.frame().EvaluateExpression(expression)
        self.assertTrue(value.IsValid(), expression + "returned a valid value")
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

    @swiftTest
    def test_simple_overload_expressions(self):
        """Test defining overloaded functions"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'Stop here to do your work.', lldb.SBFileSpec('main.swift'))

        # Here's the first function:
        value_obj = self.frame().EvaluateExpression(
            "func $overload(_ a: Int) -> Int { return 1 }\n 1")
        error = value_obj.GetError()
        self.assertSuccess(error)

        self.check_expression("$overload(10)", "1", False)

        # Here's the second function:
        value_obj = self.frame().EvaluateExpression(
            "func $overload(_ a: String) -> Int { return 2 } \n 1")
        error = value_obj.GetError()
        self.assertSuccess(error)

        self.check_expression('$overload(10)', '1', False)
        self.check_expression('$overload("some string")', '2', False)

