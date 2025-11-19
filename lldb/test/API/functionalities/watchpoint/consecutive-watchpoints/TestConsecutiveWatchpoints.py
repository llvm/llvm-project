"""
Watch contiguous memory regions with separate watchpoints, check that lldb
correctly detect which watchpoint was hit for each one.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ConsecutiveWatchpointsTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def continue_and_report_stop_reason(self, process, iter_str):
        process.Continue()
        self.assertIn(
            process.GetState(), [lldb.eStateStopped, lldb.eStateExited], iter_str
        )
        thread = process.GetSelectedThread()
        return thread.GetStopReason()

    # debugserver only gained the ability to watch larger regions
    # with this patch.
    @skipIfOutOfTreeDebugserver
    def test_consecutive_watchpoints(self):
        """Test watchpoint that covers a large region of memory."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", self.main_source_file
        )

        frame = thread.GetFrameAtIndex(0)

        field2_wp = (
            frame.locals["var"][0]
            .GetChildMemberWithName("field2")
            .Watch(True, False, True)
        )
        field3_wp = (
            frame.locals["var"][0]
            .GetChildMemberWithName("field3")
            .Watch(True, False, True)
        )
        field4_wp = (
            frame.locals["var"][0]
            .GetChildMemberWithName("field4")
            .Watch(True, False, True)
        )
        field5_wp = (
            frame.locals["var"][0]
            .GetChildMemberWithName("field5")
            .Watch(True, False, True)
        )

        # Require that the first two watchpoints
        # are set -- hopefully every machine running
        # the testsuite can support two watchpoints.
        self.assertTrue(field2_wp.IsValid())
        self.assertTrue(field3_wp.IsValid())

        reason = self.continue_and_report_stop_reason(process, "continue to field2 wp")
        self.assertEqual(reason, lldb.eStopReasonWatchpoint)
        stop_reason_watchpoint_id = (
            process.GetSelectedThread().GetStopReasonDataAtIndex(0)
        )
        self.assertEqual(stop_reason_watchpoint_id, field2_wp.GetID())

        reason = self.continue_and_report_stop_reason(process, "continue to field3 wp")
        self.assertEqual(reason, lldb.eStopReasonWatchpoint)
        stop_reason_watchpoint_id = (
            process.GetSelectedThread().GetStopReasonDataAtIndex(0)
        )
        self.assertEqual(stop_reason_watchpoint_id, field3_wp.GetID())

        # If we were able to set the second two watchpoints,
        # check that they are hit.  Some CI bots can only
        # create two watchpoints.
        if field4_wp.IsValid() and field5_wp.IsValid():
            reason = self.continue_and_report_stop_reason(
                process, "continue to field4 wp"
            )
            self.assertEqual(reason, lldb.eStopReasonWatchpoint)
            stop_reason_watchpoint_id = (
                process.GetSelectedThread().GetStopReasonDataAtIndex(0)
            )
            self.assertEqual(stop_reason_watchpoint_id, field4_wp.GetID())

            reason = self.continue_and_report_stop_reason(
                process, "continue to field5 wp"
            )
            self.assertEqual(reason, lldb.eStopReasonWatchpoint)
            stop_reason_watchpoint_id = (
                process.GetSelectedThread().GetStopReasonDataAtIndex(0)
            )
            self.assertEqual(stop_reason_watchpoint_id, field5_wp.GetID())
