"""Test that lldb command 'process signal SIGUSR1' to send a signal to the inferior works."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SendSignalTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number("main.c", "Put breakpoint here")

    @expectedFailureNetBSD(bugnumber="llvm.org/pr43959")
    @skipIfWindows  # Windows does not support signals
    def test_with_run_command(self):
        """Test that lldb command 'process signal SIGUSR1' sends a signal to the inferior process."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByLocation("main.c", self.line)
        self.assertTrue(
            breakpoint and breakpoint.GetNumLocations() == 1, VALID_BREAKPOINT
        )

        # Get the breakpoint location from breakpoint after we verified that,
        # indeed, it has one location.
        location = breakpoint.GetLocationAtIndex(0)
        self.assertTrue(location and location.IsEnabled(), VALID_BREAKPOINT_LOCATION)

        # Now launch the process, no arguments & do not stop at entry point.
        launch_info = target.GetLaunchInfo()
        launch_info.SetWorkingDirectory(self.get_process_working_directory())

        process_listener = lldb.SBListener("signal_test_listener")
        launch_info.SetListener(process_listener)
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        self.runCmd("process handle -n False -p True -s True SIGUSR1")

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid(), "We hit the first breakpoint.")

        # After resuming the process, send it a SIGUSR1 signal.

        self.setAsync(True)

        self.assertTrue(process_listener.IsValid(), "Got a good process listener")

        # Disable our breakpoint, we don't want to hit it anymore...
        breakpoint.SetEnabled(False)

        # Now continue:
        process.Continue()

        self.match_state(process_listener, lldb.eStateRunning)

        # Now signal the process, and make sure it stops:
        process.Signal(lldbutil.get_signal_number("SIGUSR1"))

        self.match_state(process_listener, lldb.eStateStopped)

        # Now make sure the thread was stopped with a SIGUSR1:
        threads = lldbutil.get_stopped_threads(process, lldb.eStopReasonSignal)
        self.assertEqual(len(threads), 1, "One thread stopped for a signal.")
        thread = threads[0]

        self.assertTrue(
            thread.GetStopReasonDataCount() >= 1, "There was data in the event."
        )
        self.assertEqual(
            thread.GetStopReasonDataAtIndex(0),
            lldbutil.get_signal_number("SIGUSR1"),
            "The stop signal was SIGUSR1",
        )

        self.match("statistics dump", [r'"signals": \[', r'"SIGUSR1": 1'])

    def match_state(self, process_listener, expected_state):
        num_seconds = 5
        broadcaster = self.process().GetBroadcaster()
        event_type_mask = lldb.SBProcess.eBroadcastBitStateChanged
        event = lldb.SBEvent()
        got_event = process_listener.WaitForEventForBroadcasterWithType(
            num_seconds, broadcaster, event_type_mask, event
        )
        self.assertTrue(got_event, "Got an event")
        state = lldb.SBProcess.GetStateFromEvent(event)
        self.assertEqual(
            state,
            expected_state,
            "It was the %s state." % lldb.SBDebugger.StateAsCString(expected_state),
        )
