# TestSwiftConditionalBreakpoint.py
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
Tests that we can set a conditional breakpoint in Swift code
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftConditionalBreakpoint(TestBase):

    @swiftTest
    def test_swift_conditional_breakpoint(self):
        """Tests that we can set a conditional breakpoint in Swift code"""
        self.build()
        self.break_commands()

    def setUp(self):
        TestBase.setUp(self)

    def check_x_and_y(self, frame, x, y):
        x_var = frame.FindVariable("x")
        y_var = frame.FindVariable("y")
        
        lldbutil.check_variable(self, x_var, value=x)
        lldbutil.check_variable(self, y_var, value=y)
        
    def break_commands(self):
        """Tests that we can set a conditional breakpoint in Swift code"""
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec("main.swift"))

        bkpt.SetCondition("x==y")
        
        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertEqual(len(threads), 1, "Hit conditional breakpoint - first time")

        self.check_x_and_y(threads[0].frames[0], '5', '5')
        
        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertEqual(len(threads), 1, "Hit conditional breakpoint - second time")

        self.check_x_and_y(threads[0].frames[0], '6', '6')

        bkpt.SetCondition('x>y')

        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertEqual(len(threads), 1, "Hit conditional breakpoint - third time")

        self.check_x_and_y(threads[0].frames[0], '3', '1')

