"""
Test that updating a breakpoint condition correctly invalidates cached state.

This test verifies that when a breakpoint condition is changed, the new condition
is properly evaluated. Previously, due to a bug in StopCondition::SetText where
the hash was computed from a moved-from string, updating conditions could fail
to invalidate cached condition state at breakpoint locations.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class UpdateBreakpointConditionTestCase(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.source = "main.c"

    @add_test_categories(["pyapi"])
    def test_update_condition_python_api(self):
        """Test that updating a breakpoint condition works correctly using Python API."""
        self.build()
        target, process, thread, breakpoint = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec(self.source)
        )

        # Set initial condition: x == y.
        breakpoint.SetCondition("x == y")
        self.assertEqual(breakpoint.GetCondition(), "x == y")

        # Need to continue since we're already stopped, but the condition wasn't set initially.
        # First hit should be at foo(5, 5) where x == y.
        process.Continue()
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "Should stop at first x == y condition")

        frame = thread.GetFrameAtIndex(0)
        x_val = frame.FindVariable("x")
        y_val = frame.FindVariable("y")
        self.assertEqual(x_val.GetValueAsSigned(), 5, "x should be 5")
        self.assertEqual(y_val.GetValueAsSigned(), 5, "y should be 5")
        self.assertEqual(breakpoint.GetHitCount(), 2)

        # Continue to second hit with x == y.
        process.Continue()
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "Should stop at second x == y condition")

        frame = thread.GetFrameAtIndex(0)
        x_val = frame.FindVariable("x")
        y_val = frame.FindVariable("y")
        self.assertEqual(x_val.GetValueAsSigned(), 6, "x should be 6")
        self.assertEqual(y_val.GetValueAsSigned(), 6, "y should be 6")
        self.assertEqual(breakpoint.GetHitCount(), 3)

        # Now update the condition to x > y.
        # This tests the fix for the bug where the hash wasn't updated correctly.
        breakpoint.SetCondition("x > y")
        self.assertEqual(breakpoint.GetCondition(), "x > y")

        # Continue - should now hit at foo(3, 1) where x > y (3 > 1).
        # Without the fix, it would incorrectly hit at foo(7, 7) due to stale condition hash.
        process.Continue()
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "Should stop at x > y condition")

        frame = thread.GetFrameAtIndex(0)
        x_val = frame.FindVariable("x")
        y_val = frame.FindVariable("y")
        self.assertEqual(x_val.GetValueAsSigned(), 3, "x should be 3")
        self.assertEqual(y_val.GetValueAsSigned(), 1, "y should be 1")
        self.assertTrue(
            x_val.GetValueAsSigned() > y_val.GetValueAsSigned(),
            "Condition x > y should be true",
        )
        self.assertEqual(breakpoint.GetHitCount(), 4)

    def test_update_condition_command(self):
        """Test that updating a breakpoint condition works correctly using breakpoint modify."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec(self.source)
        )

        # Set initial condition: x == y.
        self.runCmd("breakpoint modify -c 'x == y' 1")
        self.expect(
            "breakpoint list",
            substrs=["Condition: x == y"],
        )

        # Continue to first hit at foo(5, 5).
        self.runCmd("continue")
        self.expect("process status", PROCESS_STOPPED, patterns=["Process .* stopped"])
        self.expect(
            "frame variable x y",
            substrs=["x = 5", "y = 5"],
        )

        # Continue to second hit.
        self.runCmd("continue")
        self.expect("process status", PROCESS_STOPPED, patterns=["Process .* stopped"])
        self.expect(
            "frame variable x y",
            substrs=["x = 6", "y = 6"],
        )

        # Update condition to x > y.
        self.runCmd("breakpoint modify -c 'x > y' 1")
        self.expect(
            "breakpoint list",
            substrs=["Condition: x > y"],
        )

        # Continue - should hit at foo(3, 1) where x > y.
        self.runCmd("continue")
        self.expect("process status", PROCESS_STOPPED, patterns=["Process .* stopped"])
        self.expect(
            "frame variable x y",
            substrs=["x = 3", "y = 1"],
        )

        # Verify x > y is actually true.
        self.expect("expr x > y", substrs=["true"])
