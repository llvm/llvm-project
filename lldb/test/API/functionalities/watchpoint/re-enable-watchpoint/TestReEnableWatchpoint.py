"""
Test that a watchpoint refreshes its "current value"
when re-enabled.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ReEnableWatchpointTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def continue_and_report_stop_reason(self, process, iter_str):
        process.Continue()
        self.assertIn(
            process.GetState(), [lldb.eStateStopped, lldb.eStateExited], iter_str
        )
        thread = process.GetSelectedThread()
        return thread.GetStopReason()

    def test_reenable_watchpoint(self):
        """Test that re-enabling a watchpoint updates its saved value."""
        self.build()
        src = lldb.SBFileSpec("main.c")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break", src
        )

        #
        # Variable c has value 0.
        # Add watchpoint, check that it has cached value 0.
        #
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetValueForVariablePath("c").GetValueAsUnsigned(), 0)

        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand("watch set variable c", result)
        self.assertTrue(result.Succeeded(), "watch set ran successfully")
        results = result.GetOutput()
        self.assertIn("new value: 0", results)

        #
        # We've hit the watchpoint, confirm c has value 1 now.
        #
        reason = self.continue_and_report_stop_reason(
            process, "continue to first watchpoint"
        )
        self.assertEqual(reason, lldb.eStopReasonWatchpoint)
        self.assertEqual(frame.GetValueForVariablePath("c").GetValueAsUnsigned(), 1)

        self.runCmd("watch disable 1")

        # Continue to second breakpoint.
        # Confirm variable c has value 3.
        reason = self.continue_and_report_stop_reason(
            process, "continue to second breakpoint"
        )
        self.assertEqual(reason, lldb.eStopReasonBreakpoint)
        self.assertEqual(frame.GetValueForVariablePath("c").GetValueAsUnsigned(), 3)

        #
        # Re-enable watchpoint.
        # Confirm watchpoint has cached value 3.
        #
        self.runCmd("watch enable 1")

        result = lldb.SBCommandReturnObject()
        interp.HandleCommand("watch list", result)
        self.assertTrue(result.Succeeded(), "watch list ran successfully")
        results = result.GetOutput()
        self.assertNotIn("old value", results)
        self.assertIn("new value: 3", results)

        #
        # Resume execution and expect to hit the watchpoint.
        # Variable c now has value 4.
        #
        reason = self.continue_and_report_stop_reason(
            process, "continue to second watchpoint"
        )
        self.assertEqual(reason, lldb.eStopReasonWatchpoint)
        self.assertEqual(frame.GetValueForVariablePath("c").GetValueAsUnsigned(), 4)

        result = lldb.SBCommandReturnObject()
        interp.HandleCommand("watch list", result)
        self.assertTrue(result.Succeeded(), "watch list ran successfully")
        results = result.GetOutput()
        self.assertIn("old value: 3", results)
        self.assertIn("new value: 4", results)
