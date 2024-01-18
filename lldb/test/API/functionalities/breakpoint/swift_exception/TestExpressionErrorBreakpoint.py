# TestSwiftErrorBreakpoint.py
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
Tests catching thrown errors in using the language breakpoint
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftErrorBreakpoint(TestBase):
    @decorators.skipIfLinux  # <rdar://problem/30909618>
    @swiftTest
    def test_swift_error_no_typename(self):
        """Tests that swift error throws are correctly caught by the Swift Error breakpoint"""
        self.build()
        self.do_tests(None, True)

    @swiftTest
    def test_swift_error_matching_base_typename(self):
        """Tests that swift error throws are correctly caught by the Swift Error breakpoint"""
        self.build()
        self.do_tests("EnumError", True)

    @swiftTest
    def test_swift_error_matching_full_typename(self):
        """Tests that swift error throws are correctly caught by the Swift Error breakpoint"""
        self.build()
        self.do_tests("a.EnumError", True)

    @swiftTest
    def test_swift_error_bogus_typename(self):
        """Tests that swift error throws are correctly caught by the Swift Error breakpoint"""
        self.build()
        self.do_tests("NoSuchErrorHere", False)

    def setUp(self):
        TestBase.setUp(self)

    def do_tests(self, typename, should_stop):
        self.do_test(typename, should_stop, self.create_breakpoint_with_api)
        self.do_test(typename, should_stop, self.create_breakpoint_with_command)

    def create_breakpoint_with_api(self, target, typename):
        types = lldb.SBStringList()
        if typename:
            types.AppendString("exception-typename")
            types.AppendString(typename)
        return target.BreakpointCreateForException(
            lldb.eLanguageTypeSwift, False, True, types).GetID()

    def create_breakpoint_with_command(self, target, typename):
        return lldbutil.run_break_set_by_exception(
            self, "swift", exception_typename=typename)

    def do_test(self, typename, should_stop, make_breakpoint):
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        swift_error_bkpt_id = make_breakpoint(target, typename)
        # Note, I'm not checking locations here because we never know them
        # before launch.

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        if should_stop:
            self.assertTrue(process, PROCESS_IS_VALID)
            breakpoint_threads = lldbutil.get_threads_stopped_at_breakpoint_id(
                process, swift_error_bkpt_id)
            self.assertEqual(len(breakpoint_threads), 1,
                "We didn't stop at the error breakpoint")
        else:
            exit_state = process.GetState()
            self.assertEqual(exit_state, lldb.eStateExited,
                "We stopped at the error breakpoint when we shouldn't have.")

        target.BreakpointDelete(swift_error_bkpt_id)
