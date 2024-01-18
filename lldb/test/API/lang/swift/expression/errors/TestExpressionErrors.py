# TestExpressionErrors.py
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
Tests catching thrown errors in swift expressions
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestExpressionErrors(TestBase):
    @swiftTest
    def test_CanThrowError(self):
        """Tests that swift expressions resolve scoped variables correctly"""
        self.build()
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.target = target
        self.assertTrue(target, VALID_TARGET)

        self.checkCanThrow("IThrowObjectOver10", True)
        self.checkCanThrow("ClassError.SomeMethod", False)
 
    def checkCanThrow(self, name, expected):
        sc_list = self.target.FindFunctions(name)
        self.assertEqual(sc_list.GetSize(), 1, "Error looking for %s"%(name))
        func = sc_list[0].function
        self.assertTrue(func.IsValid(), "Couldn't find the function for %s"%(name))
        self.assertEqual(func.GetCanThrow(), expected,  "GetCanThrow was wrong for %s"%name)

    @swiftTest
    def test_swift_expression_errors(self):
        """Tests that swift expressions that throw report the errors correctly"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def continue_to_bkpt(self, process, bkpt):
        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertTrue(len(threads) == 1)

    def continue_by_pattern(self, pattern):
        bkpt = self.target.BreakpointCreateBySourceRegex(
            pattern, self.main_source_spec)
        self.assertTrue(bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)
        self.continue_to_bkpt(self.process, bkpt)
        self.target.BreakpointDelete(bkpt.GetID())

    def do_test(self):
        """Tests that swift expressions resolve scoped variables correctly"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.target = target
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        global_scope_bkpt = target.BreakpointCreateBySourceRegex(
            'Set a breakpoint here to run expressions', self.main_source_spec)
        self.assertTrue(
            global_scope_bkpt.GetNumLocations() > 0,
            VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.process = process

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, global_scope_bkpt)

        self.assertTrue(len(threads) == 1)

        options = lldb.SBExpressionOptions()
        options.SetFetchDynamicValue(lldb.eDynamicCanRunTarget)

        # FIXME: pull the "try" back out when we fix <rdar://problem/21949031>
        enum_value = self.frame().EvaluateExpression(
            "IThrowEnumOver10(101)", options)
        self.assertTrue(enum_value.IsValid(), "Got a valid enum value.")
        self.assertSuccess(enum_value.GetError(), "Error getting enum value")
        self.assertTrue(
            enum_value.GetValue() == "ImportantError",
            "Expected 'ImportantError', got %s" %
            (enum_value.GetValue()))

        object_value = self.frame().EvaluateExpression(
            "IThrowObjectOver10(101)", options)
        self.assertTrue(object_value.IsValid(), "Got a valid object value.")
        self.assertSuccess(
            object_value.GetError(),
            "Error getting object value")

        message = object_value.GetChildMemberWithName("m_message")
        self.assertTrue(message.IsValid(), "Found some m_message child.")
        self.assertSuccess(
            message.GetError(),
            "No errors fetching m_message value")
        self.assertTrue(
            message.GetSummary() == '"Over 100"',
            "Expected 'Over 100', got %s" %
            (message.GetSummary()))

