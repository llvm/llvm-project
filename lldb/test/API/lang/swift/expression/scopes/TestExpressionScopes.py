# TestExpressionScopes.py
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


class TestSwiftExpressionScopes(TestBase):
    @swiftTest
    def test_expression_scopes(self):
        """Tests that swift expressions resolve scoped variables correctly"""
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
        shadow_a_bkpt = target.BreakpointCreateBySourceRegex(
            'Shadowed in A', self.main_source_spec)
        self.assertTrue(shadow_a_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        init_bkpt = target.BreakpointCreateBySourceRegex(
            'In init.', self.main_source_spec)
        self.assertTrue(init_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.process = process

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, init_bkpt)

        self.assertTrue(len(threads) == 1)

        lldbutil.check_expression(self, self.frame(), "in_class_a", "20", use_summary=False)
        lldbutil.check_expression(self, self.frame(), "self.in_class_a", "20", use_summary=False)

        # Disable the init breakpoint so we don't have to hit it again:
        init_bkpt.SetEnabled(False)

        # Now continue to shadowed_in_a:
        self.continue_to_bkpt(process, shadow_a_bkpt)

        lldbutil.check_expression(self, self.frame(), "in_class_a", "10", use_summary=False)
        lldbutil.check_expression(self, self.frame(), "self.in_class_a", "20", use_summary=False)
        lldbutil.check_expression(self, self.frame(), "self.also_in_a", "21", use_summary=False)
        lldbutil.check_expression(self, self.frame(), "also_in_a", "21", use_summary=False)

        shadow_a_bkpt.SetEnabled(False)

        # Now set a breakpoint in the static_method and run to there:
        static_bkpt = target.BreakpointCreateBySourceRegex(
            'In class function', self.main_source_spec)
        self.assertTrue(static_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        self.continue_to_bkpt(process, static_bkpt)

        lldbutil.check_expression(self, self.frame(), "input", "10", use_summary=False)
        # This test fails on Bryce.  The self metatype doesn't generate a valid
        # type.
        lldbutil.check_expression(self, self.frame(), "self.return_ten()", "10", use_summary=False)
        static_bkpt.SetEnabled(False)

        # Now run into the setters & getters:
        set_bkpt = target.BreakpointCreateBySourceRegex(
            'In set.', self.main_source_spec)
        self.assertTrue(set_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        self.continue_to_bkpt(process, set_bkpt)
        lldbutil.check_expression(self, self.frame(), "self.backing_int", "10", use_summary=False)
        set_bkpt.SetEnabled(False)

        get_bkpt = target.BreakpointCreateBySourceRegex(
            'In get.', self.main_source_spec)
        self.assertTrue(get_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        self.continue_to_bkpt(process, get_bkpt)
        lldbutil.check_expression(self, self.frame(), "self.backing_int", "41", use_summary=False)
        get_bkpt.SetEnabled(False)

        deinit_bkpt = target.BreakpointCreateBySourceRegex(
            'In deinit.', self.main_source_spec)
        self.assertTrue(deinit_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        self.continue_to_bkpt(process, deinit_bkpt)
        lldbutil.check_expression(self, self.frame(), "self.backing_int", "41", use_summary=False)
        deinit_bkpt.SetEnabled(False)

        # Now let's try the subscript getter & make sure that that works:

        str_sub_get_bkpt = target.BreakpointCreateBySourceRegex(
            'In string subscript getter', self.main_source_spec)
        self.assertTrue(
            str_sub_get_bkpt.GetNumLocations() > 0,
            VALID_BREAKPOINT)

        self.continue_to_bkpt(process, str_sub_get_bkpt)
        lldbutil.check_expression(self, self.frame(), "self.backing_int", "10", use_summary=False)

        str_sub_get_bkpt.SetEnabled(False)

        # Next run into the closure that captures self and make sure that
        # works:
        closure_bkpt = target.BreakpointCreateBySourceRegex(
            'Break here in closure', self.main_source_spec)
        self.assertTrue(closure_bkpt.GetNumLocations() > 0, VALID_BREAKPOINT)

        self.continue_to_bkpt(process, closure_bkpt)
        lldbutil.check_expression(self, self.frame(), "self.backing_int", "10", use_summary=False)
        lldbutil.check_expression(self, self.frame(), "a_string", '"abcde"', use_summary=True)

        # Now set a breakpoint in the struct method and run to there:
        self.continue_by_pattern('Break here in struct')
        lldbutil.check_expression(self, self.frame(), "a", "\"foo\"", use_summary=True)
        lldbutil.check_expression(self, self.frame(), "self.b", "5", use_summary=False)
