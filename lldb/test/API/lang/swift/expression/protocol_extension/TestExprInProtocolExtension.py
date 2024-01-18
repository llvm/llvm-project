# TestExprInProtocolExtension.py
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
Tests scoped variables with swift expressions
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftExprInProtocolExtension(TestBase):
    def continue_to_bkpt(self, process, bkpt):
        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertTrue(len(threads) == 1)

    def continue_by_pattern(self, pattern):
        bkpt = self.target.BreakpointCreateBySourceRegex(
            pattern, self.main_source_spec)
        self.assertTrue(bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)
        self.continue_to_bkpt(self.process, bkpt)
        self.target.BreakpointDelete(bkpt.GetID())

    @swiftTest
    def test_protocol_extension(self):
        """Tests that swift expressions in protocol extension functions behave correctly"""
        self.build()

        # Create the target
        target = self.dbg.CreateTarget(self.getBuildArtifact())
        self.target = target
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        static_bkpt = target.BreakpointCreateBySourceRegex(
            'break here in static func', lldb.SBFileSpec('main.swift'))
        self.assertTrue(static_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        method_bkpt = target.BreakpointCreateBySourceRegex(
            'break here in method', lldb.SBFileSpec('main.swift'))
        self.assertTrue(method_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.process = process
        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, method_bkpt)

        self.assertTrue(len(threads) == 1)

        # Check that we can evaluate expressions correctly in the struct
        # method.
        lldbutil.check_expression(self, self.frame(), "self.x", "10", False)
        lldbutil.check_expression(self, self.frame(), "self.y", '"Hello world"', True)
        lldbutil.check_expression(self, self.frame(), "local_var", "111", False)

        # And check that we got the type of self right:
        self_var = self.frame().EvaluateExpression(
            "self", lldb.eDynamicCanRunTarget)
        self_type_name = self_var.GetTypeName()
        print("Self type name is: ", self_type_name)

        # Not checking yet since we don't get this right.

        # Now continue to the static method and check things there:
        self.continue_to_bkpt(process, static_bkpt)

        lldbutil.check_expression(self, self.frame(), "self.cvar", "333", False)
        lldbutil.check_expression(self, self.frame(), "local_var", "222", False)

        # This continues to the class version:
        self.continue_to_bkpt(process, method_bkpt)
        # Check that we can evaluate expressions correctly in the struct
        # method.
        lldbutil.check_expression(self, self.frame(), "self.x", "10", False)
        lldbutil.check_expression(self, self.frame(), "self.y", '"Hello world"', True)
        lldbutil.check_expression(self, self.frame(), "local_var", "111", False)

        # And check that we got the type of self right:
        self_var = self.frame().EvaluateExpression(
            "self", lldb.eDynamicCanRunTarget)
        self_type_name = self_var.GetTypeName()
        print("Self type name is: ", self_type_name)

        # Not checking yet since we don't get this right.

        # Now continue to the static method and check things there:
        self.continue_to_bkpt(process, static_bkpt)

        lldbutil.check_expression(self, self.frame(), "self.cvar", "333", False)
        lldbutil.check_expression(self, self.frame(), "local_var", "222", False)

        # This continues to the enum version:
        self.continue_to_bkpt(process, method_bkpt)
        # Check that we can evaluate expressions correctly in the struct
        # method.
        lldbutil.check_expression(self, self.frame(), "self.x", "10", False)
        lldbutil.check_expression(self, self.frame(), "self.y", '"Hello world"', True)
        lldbutil.check_expression(self, self.frame(), "local_var", "111", False)

        # And check that we got the type of self right:
        self_var = self.frame().EvaluateExpression(
            "self", lldb.eDynamicCanRunTarget)
        self_type_name = self_var.GetTypeName()
        print("Self type name is: ", self_type_name)

        # Not checking yet since we don't get this right.

        # Now continue to the static method and check things there:
        self.continue_to_bkpt(process, static_bkpt)

        lldbutil.check_expression(self, self.frame(), "self.cvar", "333", False)
        lldbutil.check_expression(self, self.frame(), "local_var", "222", False)
