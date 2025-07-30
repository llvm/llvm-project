import lldb
import time
import unittest
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.gdbclientutils import *
from lldbsuite.test.lldbreverse import ReverseTestBase
from lldbsuite.test import lldbutil


class TestReverseContinueBreakpoints(ReverseTestBase):
    @skipIfRemote
    def test_reverse_continue(self):
        self.reverse_continue_internal(async_mode=False)

    @skipIfRemote
    def test_reverse_continue_async(self):
        self.reverse_continue_internal(async_mode=True)

    def reverse_continue_internal(self, async_mode):
        target, process, initial_threads = self.setup_recording(async_mode)

        # Reverse-continue. We'll stop at the point where we started recording.
        status = process.ContinueInDirection(lldb.eRunReverse)
        self.assertSuccess(status)
        self.expect_async_state_changes(
            async_mode, process, [lldb.eStateRunning, lldb.eStateStopped]
        )
        self.expect(
            "thread list",
            STOPPED_DUE_TO_HISTORY_BOUNDARY,
            substrs=["stopped", "stop reason = history boundary"],
        )

        # Continue forward normally until the target exits.
        status = process.ContinueInDirection(lldb.eRunForward)
        self.expect_async_state_changes(
            async_mode, process, [lldb.eStateRunning, lldb.eStateExited]
        )
        self.assertSuccess(status)
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

    @skipIfRemote
    def test_reverse_continue_breakpoint(self):
        self.reverse_continue_breakpoint_internal(async_mode=False)

    @skipIfRemote
    def test_reverse_continue_breakpoint_async(self):
        self.reverse_continue_breakpoint_internal(async_mode=True)

    def reverse_continue_breakpoint_internal(self, async_mode):
        target, process, initial_threads = self.setup_recording(async_mode)

        # Reverse-continue to the function "trigger_breakpoint".
        trigger_bkpt = target.BreakpointCreateByName("trigger_breakpoint", None)
        status = process.ContinueInDirection(lldb.eRunReverse)
        self.assertSuccess(status)
        self.expect_async_state_changes(
            async_mode, process, [lldb.eStateRunning, lldb.eStateStopped]
        )
        threads_now = lldbutil.get_threads_stopped_at_breakpoint(process, trigger_bkpt)
        self.assertEqual(threads_now, initial_threads)

    @skipIfRemote
    @skipIf(
        oslist=["windows"],
        archs=["x86_64"],
        bugnumber="github.com/llvm/llvm-project/issues/138084",
    )
    def test_reverse_continue_skip_breakpoint(self):
        self.reverse_continue_skip_breakpoint_internal(async_mode=False)

    @skipIfRemote
    @skipIf(
        oslist=["windows"],
        archs=["x86_64"],
        bugnumber="github.com/llvm/llvm-project/issues/138084",
    )
    def test_reverse_continue_skip_breakpoint_async(self):
        self.reverse_continue_skip_breakpoint_internal(async_mode=True)

    def reverse_continue_skip_breakpoint_internal(self, async_mode):
        target, process, initial_threads = self.setup_recording(async_mode)

        # Reverse-continue over a breakpoint at "trigger_breakpoint" whose
        # condition is false (via function call).
        # This tests that we continue in the correct direction after hitting
        # the breakpoint.
        trigger_bkpt = target.BreakpointCreateByName("trigger_breakpoint", None)
        trigger_bkpt.SetCondition("false_condition()")
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

    @skipIfRemote
    def test_continue_preserves_direction(self):
        self.continue_preserves_direction_internal(async_mode=False)

    @skipIfRemote
    def test_continue_preserves_direction_asyhc(self):
        self.continue_preserves_direction_internal(async_mode=True)

    def continue_preserves_direction_internal(self, async_mode):
        target, process, initial_threads = self.setup_recording(async_mode)

        # Reverse-continue to the function "trigger_breakpoint".
        trigger_bkpt = target.BreakpointCreateByName("trigger_breakpoint", None)
        status = process.ContinueInDirection(lldb.eRunReverse)
        self.assertSuccess(status)
        self.expect_async_state_changes(
            async_mode, process, [lldb.eStateRunning, lldb.eStateStopped]
        )
        # This should continue in reverse.
        status = process.Continue()
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
        self.start_recording()
        stop_recording_bkpt = target.BreakpointCreateByName("stop_recording", None)
        self.assertTrue(stop_recording_bkpt.GetNumLocations() > 0)
        lldbutil.continue_to_breakpoint(process, stop_recording_bkpt)
        target.BreakpointDelete(stop_recording_bkpt.GetID())
        self.stop_recording()

        self.dbg.SetAsync(async_mode)
        self.expect_async_state_changes(async_mode, process, [lldb.eStateStopped])

        return target, process, initial_threads

    def expect_async_state_changes(self, async_mode, process, states):
        if not async_mode:
            return
        listener = self.dbg.GetListener()
        lldbutil.expect_state_changes(self, listener, process, states)
