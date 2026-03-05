"""
Test that the run locker really does work to keep
us from running SB API that should only be run
while stopped.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestRunLocker(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows # Windows doesn't have unistd.h
    # Is flaky on Linux AArch64 buildbot.
    @skipIf(oslist=["linux"], archs=["aarch64"])
    def test_run_locker(self):
        """Test that the run locker is set correctly when we launch"""
        self.build()
        self.runlocker_test(False)

    @skipIfWindows # Windows doesn't have unistd.h
    # Is flaky on Linux AArch64 buildbot.
    @skipIf(oslist=["linux"], archs=["aarch64"])
    def test_run_locker_stop_at_entry(self):
        """Test that the run locker is set correctly when we launch"""
        self.build()
        self.runlocker_test(False)

    @skipIfWindows # Windows doesn't have unistd.h
    def test_during_breakpoint_command(self):
        """Test that other threads don't see the process as stopped
        until we actually finish running the breakpoint callback."""
        self.build()
        self.during_breakpoint_command()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.main_source_file = lldb.SBFileSpec("main.c")

    def start_process(self, flags, bkpt_text):
        """Start up an async process, using flags for the launch flags
        if it is not None.  Also, set a breakpoint before running using
        bkpt_text as the source regex.  Returns the target, process, listener
        and breakpoint."""

        target = lldbutil.run_to_breakpoint_make_target(self)

        bkpt = None
        if bkpt_text:
            bkpt = target.BreakpointCreateBySourceRegex(
                bkpt_text, self.main_source_file
            )

        launch_info = target.GetLaunchInfo()
        if flags:
            flags = launch_info.GetFlags()
            launch_info.SetFlags(flags | flags)

        error = lldb.SBError()
        # We are trying to do things when the process is running, so
        # we have to run the debugger asynchronously.
        self.dbg.SetAsync(True)

        listener = lldb.SBListener("test-run-lock-listener")
        launch_info.SetListener(listener)
        process = target.Launch(launch_info, error)
        self.assertSuccess(error, "Launched the process")

        event = lldb.SBEvent()

        event_result = listener.WaitForEvent(10, event)
        self.assertTrue(event_result, "timed out waiting for launch")
        state_type = lldb.SBProcess.GetStateFromEvent(event)
        # We don't always see a launching...
        if state_type == lldb.eStateLaunching:
            event_result = listener.WaitForEvent(10, event)
            self.assertTrue(
                event_result, "Timed out waiting for running after launching"
            )
            state_type = lldb.SBProcess.GetStateFromEvent(event)

        self.assertState(state_type, lldb.eStateRunning, "Didn't get a running event")
        return (target, process, listener, bkpt)

    def try_expression(self, target):
        val = target.EvaluateExpression("SomethingToCall()")
        # There was a bug [#93313] in the printing that would cause repr to crash, so I'm
        # testing that separately.
        self.assertIn(
            "can't evaluate expressions when the process is running",
            repr(val),
            "repr works"
        )
        error = val.GetError()
        self.assertTrue(error.Fail(), "Failed to run expression")
        self.assertIn(
            "can't evaluate expressions when the process is running",
            error.GetCString(),
            "Stopped by stop locker",
        )

        # This should also fail if we try to use the script interpreter directly:
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        ret = interp.HandleCommand(
            "script var = lldb.frame.EvaluateExpression('SomethingToCall()'); var.GetError().GetCString()",
            result,
        )
        self.assertIn(
            "can't evaluate expressions when the process is running", result.GetOutput()
        )

    def runlocker_test(self, stop_at_entry):
        """The code to stop at entry handles events slightly differently, so we test both versions of process launch."""
        flags = None

        if stop_at_entry:
            flags = lldb.eLaunchFlagsStopAtEntry

        target, process, listener, _ = self.start_process(flags, None)

        # We aren't checking the entry state, but just making sure
        # the running state is set properly if we continue in this state.

        if stop_at_entry:
            event_result = listener.WaitForEvent(10, event)
            self.assertTrue(event_result, "Timed out waiting for stop at entry stop")
            state_type = lldb.SBProcess.GetStateFromEvent(event)
            self.assertState(state_type, eStateStopped, "Stop at entry stopped")
            process.Continue()

        # Okay, now the process is running, make sure we can't do things
        # you aren't supposed to do while running, and that we get some
        # actual error:
        self.try_expression(target)

    def during_breakpoint_command(self):
        target, process, listener, bkpt = self.start_process(None, "sleep.1.")
        # The process should be stopped at our breakpoint, wait for that.
        event = lldb.SBEvent()
        result = listener.WaitForEvent(10, event)
        self.assertTrue(event.IsValid(), "Didn't time out waiting for breakpoint")
        state_type = lldb.SBProcess.GetStateFromEvent(event)
        self.assertState(state_type, lldb.eStateStopped, "Didn't get a stopped event")

        # Now add a breakpoint callback that will stall for a while, and we'll wait a
        # much shorter interval and each time we wake up ensure that we still see the
        # process as running, and can't do things we aren't allowed to in that state.
        commands = (
            "import time;print('About to sleep');time.sleep(20);print('Done sleeping')"
        )

        bkpt.SetScriptCallbackBody(commands)

        process.Continue()
        result = listener.WaitForEvent(10, event)
        state_type = lldb.SBProcess.GetStateFromEvent(event)
        self.assertState(state_type, lldb.eStateRunning, "We started running")
        counter = 0
        while state_type == lldb.eStateRunning:
            print("About to wait")
            result = listener.GetNextEvent(event)
            counter += 1
            print(f"Woke up {counter} times")
            if not result:
                self.try_expression(target)
            else:
                state_type = lldb.SBProcess.GetStateFromEvent(event)
