"""
Test that the run locker really does work to keep
us from running SB API that should only be run
while stopped.  This test is mostly concerned with
what happens between launch and first stop.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestRunLocker(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(oslist=["windows"])
    # Is flaky on Linux AArch64 buildbot.
    @skipIf(oslist=["linux"], archs=["aarch64"])
    def test_run_locker(self):
        """Test that the run locker is set correctly when we launch"""
        self.build()
        self.runlocker_test(False)

    @expectedFailureAll(oslist=["windows"])
    # Is flaky on Linux AArch64 buildbot.
    @skipIf(oslist=["linux"], archs=["aarch64"])
    def test_run_locker_stop_at_entry(self):
        """Test that the run locker is set correctly when we launch"""
        self.build()
        self.runlocker_test(False)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.main_source_file = lldb.SBFileSpec("main.c")

    def runlocker_test(self, stop_at_entry):
        """The code to stop at entry handles events slightly differently, so
        we test both versions of process launch."""

        target = lldbutil.run_to_breakpoint_make_target(self)

        launch_info = target.GetLaunchInfo()
        if stop_at_entry:
            flags = launch_info.GetFlags()
            launch_info.SetFlags(flags | lldb.eLaunchFlagStopAtEntry)

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
