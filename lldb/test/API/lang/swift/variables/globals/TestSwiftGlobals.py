# TestSwiftGlobals.py
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
Check that we can examine module globals in the expression parser.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftGlobals(TestBase):
    @swiftTest
    def test_swift_globals(self):
        """Check that we can examine module globals in the expression parser"""
        self.build()
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Target variables. This is not actually expected to work, but
        # also shouldn't crash.
        g_counter = target.EvaluateExpression("g_counter")
        self.assertTrue(
            g_counter.IsValid(),
            "g_counter returned a valid value object.")

        # Set the breakpoints
        main_source_spec = lldb.SBFileSpec('main.swift')
        outer_bkpt = target.BreakpointCreateBySourceRegex(
            'Set top_level breakpoint here', main_source_spec)
        self.assertGreater(outer_bkpt.GetNumLocations(), 0, VALID_BREAKPOINT)

        function_bkpt = target.BreakpointCreateBySourceRegex(
            'Set function breakpoint here', main_source_spec)
        self.assertGreater(function_bkpt.GetNumLocations(), 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, outer_bkpt)

        self.assertEqual(len(threads), 1)
        self.thread = threads[0]

        # All the variables should be uninitialized at this point.  Maybe sure
        # they look that way:
        frame = self.thread.frames[0]
        options = lldb.SBExpressionOptions()
        options.SetFetchDynamicValue(lldb.eDynamicCanRunTarget)

        error = lldb.SBError()

        # Examine the variables before initialization:

        g_counter = frame.EvaluateExpression("g_counter", options)
        self.assertTrue(
            g_counter.IsValid(),
            "g_counter returned a valid value object.")
        value = g_counter.GetValueAsSigned(error)
        self.assertSuccess(error, "Got a value for g_counter")
        self.assertEqual(value, 0,
            "g_counter value is the uninitialized one.")

        foo_var = frame.EvaluateExpression("my_foo", options)
        self.assertTrue(
            foo_var.IsValid(),
            "foo_var returned a valid value object.")
        value = foo_var.GetValueAsUnsigned(error)
        self.assertSuccess(error, "foo_var has a value.")
        self.assertEqual(value, 0, "foo_var is null before initialization.")

        my_large_dude = frame.EvaluateExpression("my_large_dude", options)
        self.assertTrue(my_large_dude.IsValid(),
                        "my_large_dude returned a valid value object.")
        value = my_large_dude.GetValue()
        self.assertSuccess(error, "Got a value for my_large_dude")
        self.assertIsNone(
            value,
            "my_large_dude value is the uninitialized one.")

        # Now proceed to the breakpoint in our main function, make sure we can
        # still read these variables and they now have the right values.
        threads = lldbutil.continue_to_breakpoint(process, function_bkpt)
        self.assertEqual(len(threads), 1)

        self.thread = threads[0]

        # Examine the variables before initialization:

        g_counter = frame.EvaluateExpression("g_counter", options)
        self.assertTrue(
            g_counter.IsValid(),
            "g_counter returned a valid value object.")
        value = g_counter.GetValueAsSigned(error)
        self.assertSuccess(error, "Got a value for g_counter")
        self.assertTrue(value == 2, "g_counter value should be 2.")

        foo_var = frame.EvaluateExpression("my_foo", options)
        self.assertTrue(
            foo_var.IsValid(),
            "foo_var returned a valid value object.")
        foo_var_x = foo_var.GetChildMemberWithName("x")
        self.assertTrue(foo_var_x.IsValid(), "Got value object for foo_var.x")
        value = foo_var_x.GetValueAsUnsigned(error)
        self.assertSuccess(error, "foo_var.x has a value.")
        self.assertEqual(value, 1, "foo_var is null before initialization.")

        my_large_dude = frame.EvaluateExpression("my_large_dude", options)
        self.assertTrue(my_large_dude.IsValid(),
                        "my_large_dude returned a valid value object.")
        my_large_dude_y = my_large_dude.GetChildMemberWithName("y")
        self.assertTrue(
            my_large_dude_y.IsValid(),
            "Got value object for my_large_dude.y")
        value = my_large_dude_y.GetValueAsUnsigned(error)
        self.assertSuccess(error, "Got a value for my_large_dude.y")
        self.assertEqual(
            value, 20,
            "my_large_dude value is the uninitialized one.")

