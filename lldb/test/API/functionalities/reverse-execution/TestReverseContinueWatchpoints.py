import lldb
import time
import unittest
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbreverse import ReverseTestBase
from lldbsuite.test import lldbutil


class TestReverseContinueWatchpoints(ReverseTestBase):
    @skipIfRemote
    # Watchpoints don't work in single-step mode
    @skipIf(macos_version=["<", "15.0"], archs=["arm64"])
    def test_reverse_continue_watchpoint(self):
        self.reverse_continue_watchpoint_internal(async_mode=False)

    @skipIfRemote
    # Watchpoints don't work in single-step mode
    @skipIf(macos_version=["<", "15.0"], archs=["arm64"])
    def test_reverse_continue_watchpoint_async(self):
        self.reverse_continue_watchpoint_internal(async_mode=True)

    def reverse_continue_watchpoint_internal(self, async_mode):
        target, process, initial_threads, watch_addr = self.setup_recording(async_mode)

        error = lldb.SBError()
        wp_opts = lldb.SBWatchpointOptions()
        wp_opts.SetWatchpointTypeWrite(lldb.eWatchpointWriteTypeOnModify)
        watchpoint = target.WatchpointCreateByAddress(watch_addr, 4, wp_opts, error)
        self.assertTrue(watchpoint)

        watch_var = target.EvaluateExpression("*g_watched_var_ptr")
        self.assertEqual(watch_var.GetValueAsSigned(-1), 2)

        # Reverse-continue to the function "trigger_watchpoint".
        status = process.ContinueInDirection(lldb.eRunReverse)
        self.assertSuccess(status)
        self.expect_async_state_changes(
            async_mode, process, [lldb.eStateRunning, lldb.eStateStopped]
        )
        # We should stop at the point just before the location was modified.
        watch_var = target.EvaluateExpression("*g_watched_var_ptr")
        self.assertEqual(watch_var.GetValueAsSigned(-1), 1)
        self.expect(
            "thread list",
            STOPPED_DUE_TO_WATCHPOINT,
            substrs=["stopped", "trigger_watchpoint", "stop reason = watchpoint 1"],
        )

        # Stepping forward one instruction should change the value of the variable.
        initial_threads[0].StepInstruction(False)
        self.expect_async_state_changes(
            async_mode, process, [lldb.eStateRunning, lldb.eStateStopped]
        )
        watch_var = target.EvaluateExpression("*g_watched_var_ptr")
        self.assertEqual(watch_var.GetValueAsSigned(-1), 2)
        self.expect(
            "thread list",
            STOPPED_DUE_TO_WATCHPOINT,
            substrs=["stopped", "trigger_watchpoint", "stop reason = watchpoint 1"],
        )

    @skipIfRemote
    # Watchpoints don't work in single-step mode
    @skipIf(macos_version=["<", "15.0"], archs=["arm64"])
    @skipIf(
        oslist=["windows"],
        archs=["x86_64"],
        bugnumber="github.com/llvm/llvm-project/issues/138084",
    )
    def test_reverse_continue_skip_watchpoint(self):
        self.reverse_continue_skip_watchpoint_internal(async_mode=False)

    @skipIfRemote
    # Watchpoints don't work in single-step mode
    @skipIf(macos_version=["<", "15.0"], archs=["arm64"])
    @skipIf(
        oslist=["windows"],
        archs=["x86_64"],
        bugnumber="github.com/llvm/llvm-project/issues/138084",
    )
    def test_reverse_continue_skip_watchpoint_async(self):
        self.reverse_continue_skip_watchpoint_internal(async_mode=True)

    def reverse_continue_skip_watchpoint_internal(self, async_mode):
        target, process, initial_threads, watch_addr = self.setup_recording(async_mode)

        # Reverse-continue over a watchpoint whose condition is false
        # (via function call).
        # This tests that we continue in the correct direction after hitting
        # the watchpoint.
        error = lldb.SBError()
        wp_opts = lldb.SBWatchpointOptions()
        wp_opts.SetWatchpointTypeWrite(lldb.eWatchpointWriteTypeOnModify)
        watchpoint = target.WatchpointCreateByAddress(watch_addr, 4, wp_opts, error)
        self.assertTrue(watchpoint)
        watchpoint.SetCondition("false_condition()")
        status = process.ContinueInDirection(lldb.eRunReverse)
        self.expect_async_state_changes(
            async_mode, process, [lldb.eStateRunning, lldb.eStateStopped]
        )
        self.assertSuccess(status)
        self.expect(
            "thread list",
            STOPPED_DUE_TO_HISTORY_BOUNDARY,
            substrs=["stopped", "stop reason = history boundary"],
        )

    def setup_recording(self, async_mode):
        """
        Record execution of code between "start_recording" and "stop_recording" breakpoints.

        Returns with the target stopped at "stop_recording", with recording disabled,
        ready to reverse-execute.
        """
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        process = self.connect(target)

        # Record execution from the start of the function "start_recording"
        # to the start of the function "stop_recording". We want to keep the
        # interval that we record as small as possible to minimize the run-time
        # of our single-stepping recorder.
        start_recording_bkpt = target.BreakpointCreateByName("start_recording", None)
        self.assertTrue(start_recording_bkpt.GetNumLocations() > 0)
        initial_threads = lldbutil.continue_to_breakpoint(process, start_recording_bkpt)
        self.assertEqual(len(initial_threads), 1)
        target.BreakpointDelete(start_recording_bkpt.GetID())

        frame0 = initial_threads[0].GetFrameAtIndex(0)
        watched_var_ptr = frame0.FindValue(
            "g_watched_var_ptr", lldb.eValueTypeVariableGlobal
        )
        watch_addr = watched_var_ptr.GetValueAsUnsigned(0)
        self.assertTrue(watch_addr > 0)

        self.start_recording()
        stop_recording_bkpt = target.BreakpointCreateByName("stop_recording", None)
        self.assertTrue(stop_recording_bkpt.GetNumLocations() > 0)
        lldbutil.continue_to_breakpoint(process, stop_recording_bkpt)
        target.BreakpointDelete(stop_recording_bkpt.GetID())
        self.stop_recording()

        self.dbg.SetAsync(async_mode)
        self.expect_async_state_changes(async_mode, process, [lldb.eStateStopped])

        return target, process, initial_threads, watch_addr

    def expect_async_state_changes(self, async_mode, process, states):
        if not async_mode:
            return
        listener = self.dbg.GetListener()
        lldbutil.expect_state_changes(self, listener, process, states)
