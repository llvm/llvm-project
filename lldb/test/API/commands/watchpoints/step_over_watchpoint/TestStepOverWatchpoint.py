"""Test stepping over watchpoints and instruction stepping past watchpoints."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestStepOverWatchpoint(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def get_to_start(self, bkpt_text):
        """Test stepping over watchpoints and instruction stepping past watchpoints.."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, bkpt_text, lldb.SBFileSpec("main.c")
        )
        return (target, process, thread, frame, read_watchpoint)

    @add_test_categories(["basic_process"])
    @expectedFailureAll(
        macos_version=["<", "14.4"],
        archs=["aarch64", "arm"],
        bugnumber="<rdar://problem/106868647>",
    )
    def test_step_over_read_watchpoint(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here for read watchpoints", lldb.SBFileSpec("main.c")
        )

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Failed to get frame.")

        read_value = frame.FindValue("g_watch_me_read", lldb.eValueTypeVariableGlobal)
        self.assertTrue(read_value.IsValid(), "Failed to find read value.")

        error = lldb.SBError()

        # resolve_location=True, read=True, write=False
        read_watchpoint = read_value.Watch(True, True, False, error)
        self.assertSuccess(error, "Error while setting watchpoint")
        self.assertTrue(read_watchpoint, "Failed to set read watchpoint.")

        # Disable the breakpoint we hit so we don't muddy the waters with
        # stepping off from the breakpoint:
        bkpt.SetEnabled(False)

        thread.StepOver()
        self.assertStopReason(
            thread.GetStopReason(),
            lldb.eStopReasonWatchpoint,
            STOPPED_DUE_TO_WATCHPOINT,
        )
        self.assertEqual(thread.GetStopDescription(20), "watchpoint 1")

        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        self.assertEqual(thread.GetStopDescription(20), "step over")

        self.step_inst_for_watchpoint(1)

    @add_test_categories(["basic_process"])
    @expectedFailureAll(
        macos_version=["<", "14.4"],
        archs=["aarch64", "arm"],
        bugnumber="<rdar://problem/106868647>",
    )
    def test_step_over_write_watchpoint(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here for modify watchpoints", lldb.SBFileSpec("main.c")
        )

        # Disable the breakpoint we hit so we don't muddy the waters with
        # stepping off from the breakpoint:
        bkpt.SetEnabled(False)

        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid(), "Failed to get frame.")

        write_value = frame.FindValue("g_watch_me_write", lldb.eValueTypeVariableGlobal)
        self.assertTrue(write_value, "Failed to find write value.")

        error = lldb.SBError()
        # resolve_location=True, read=False, modify=True
        write_watchpoint = write_value.Watch(True, False, True, error)
        self.assertTrue(write_watchpoint, "Failed to set write watchpoint.")
        self.assertSuccess(error, "Error while setting watchpoint")

        thread.StepOver()
        self.assertStopReason(
            thread.GetStopReason(),
            lldb.eStopReasonWatchpoint,
            STOPPED_DUE_TO_WATCHPOINT,
        )
        self.assertEqual(thread.GetStopDescription(20), "watchpoint 1")

        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        self.assertEqual(thread.GetStopDescription(20), "step over")

        self.step_inst_for_watchpoint(1)

    def step_inst_for_watchpoint(self, wp_id):
        watchpoint_hit = False
        current_line = self.frame().GetLineEntry().GetLine()
        while self.frame().GetLineEntry().GetLine() == current_line:
            self.thread().StepInstruction(False)  # step_over=False
            stop_reason = self.thread().GetStopReason()
            if stop_reason == lldb.eStopReasonWatchpoint:
                self.assertFalse(watchpoint_hit, "Watchpoint already hit.")
                expected_stop_desc = "watchpoint %d" % wp_id
                actual_stop_desc = self.thread().GetStopDescription(20)
                self.assertEqual(
                    actual_stop_desc, expected_stop_desc, "Watchpoint ID didn't match."
                )
                watchpoint_hit = True
            else:
                self.assertStopReason(
                    stop_reason, lldb.eStopReasonPlanComplete, STOPPED_DUE_TO_STEP_IN
                )
        self.assertTrue(watchpoint_hit, "Watchpoint never hit.")
