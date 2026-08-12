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

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.main_source_file = lldb.SBFileSpec("main.c")

    @expectedFailureAll(oslist=["windows"])
    def test_run_locker(self):
        """Test that the run locker is set correctly as we're running"""
        self.build()
        target = lldbutil.run_to_breakpoint_make_target(self)
        launch_info = target.GetLaunchInfo()

        error = lldb.SBError()
        # We are trying to do things when the process is running, so
        # we have to run the debugger asynchronously.
        self.dbg.SetAsync(True)

        main_bp = target.BreakpointCreateByName("main")

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

        # We may be in eStateStopped if we hit the breakpoint already.
        if state_type != lldb.eStateStopped:
            self.assertState(
                state_type, lldb.eStateRunning, "Didn't get a running event"
            )
            event_result = listener.WaitForEvent(10, event)
            self.assertTrue(event_result, "timed out waiting for breakpoint stop")
            state_type = lldb.SBProcess.GetStateFromEvent(event)

        self.assertState(state_type, lldb.eStateStopped, "Stop at main stopped")
        main_bp.SetEnabled(False)
        process.Continue()

        event_result = listener.WaitForEvent(10, event)
        self.assertTrue(event_result, "timed out waiting for process resume")
        state_type = lldb.SBProcess.GetStateFromEvent(event)

        self.assertState(state_type, lldb.eStateRunning, "Continue after main() bp")

        # Okay, now the process is running, make sure we can't do things
        # you aren't supposed to do while running, and that we get some
        # actual error:
        val = target.EvaluateExpression("SomethingToCall()")
        # There was a bug [#93313] in the printing that would cause repr to crash, so I'm
        # testing that separately.
        self.assertIn(
            "can't evaluate expressions when the process is running",
            repr(val),
            "repr works",
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
            "script var = lldb.target.EvaluateExpression('SomethingToCall()'); var.GetError().GetCString()",
            result,
        )
        self.assertIn(
            "can't evaluate expressions when the process is running", result.GetOutput()
        )
