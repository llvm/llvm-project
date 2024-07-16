"""
Test lldb Python event APIs.
"""

import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import random

@skipIfLinux  # llvm.org/pr25924, sometimes generating SIGSEGV
class EventAPITestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line = line_number(
            "main.c", '// Find the line number of function "c" here.'
        )
        random.seed()

    @expectedFailureAll(
        oslist=["linux"], bugnumber="llvm.org/pr23730 Flaky, fails ~1/10 cases"
    )
    @skipIfWindows  # This is flakey on Windows AND when it fails, it hangs: llvm.org/pr38373
    @skipIfNetBSD
    def test_listen_for_and_print_event(self):
        """Exercise SBEvent API."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        self.dbg.SetAsync(True)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName("c", "a.out")

        listener = lldb.SBListener("my listener")

        # Now launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        flags = target.GetLaunchInfo().GetLaunchFlags()
        process = target.Launch(
            listener,
            None,  # argv
            None,  # envp
            None,  # stdin_path
            None,  # stdout_path
            None,  # stderr_path
            None,  # working directory
            flags,  # launch flags
            False,  # Stop at entry
            error,
        )  # error

        self.assertEqual(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        # Create an empty event object.
        event = lldb.SBEvent()

        traceOn = self.TraceOn()
        if traceOn:
            lldbutil.print_stacktraces(process)

        # Create MyListeningThread class to wait for any kind of event.
        import threading

        class MyListeningThread(threading.Thread):
            def run(self):
                count = 0
                # Let's only try at most 4 times to retrieve any kind of event.
                # After that, the thread exits.
                while not count > 3:
                    if traceOn:
                        print("Try wait for event...")
                    if listener.WaitForEvent(5, event):
                        if traceOn:
                            desc = lldbutil.get_description(event)
                            print("Event description:", desc)
                            print("Event data flavor:", event.GetDataFlavor())
                            print(
                                "Process state:",
                                lldbutil.state_type_to_str(process.GetState()),
                            )
                            print()
                    else:
                        if traceOn:
                            print("timeout occurred waiting for event...")
                    count = count + 1
                listener.Clear()
                return

        # Let's start the listening thread to retrieve the events.
        my_thread = MyListeningThread()
        my_thread.start()

        # Use Python API to continue the process.  The listening thread should be
        # able to receive the state changed events.
        process.Continue()

        # Use Python API to kill the process.  The listening thread should be
        # able to receive the state changed event, too.
        process.Kill()

        # Wait until the 'MyListeningThread' terminates.
        my_thread.join()

        # Shouldn't we be testing against some kind of expectation here?

    @expectedFlakeyLinux("llvm.org/pr23730")  # Flaky, fails ~1/100 cases
    @skipIfWindows  # This is flakey on Windows AND when it fails, it hangs: llvm.org/pr38373
    @skipIfNetBSD
    def test_wait_for_event(self):
        """Exercise SBListener.WaitForEvent() API."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        self.dbg.SetAsync(True)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName("c", "a.out")
        self.trace("breakpoint:", breakpoint)
        self.assertTrue(
            breakpoint and breakpoint.GetNumLocations() == 1, VALID_BREAKPOINT
        )

        # Get the debugger listener.
        listener = self.dbg.GetListener()

        # Now launch the process, and do not stop at entry point.
        error = lldb.SBError()
        flags = target.GetLaunchInfo().GetLaunchFlags()
        process = target.Launch(
            listener,
            None,  # argv
            None,  # envp
            None,  # stdin_path
            None,  # stdout_path
            None,  # stderr_path
            None,  # working directory
            flags,  # launch flags
            False,  # Stop at entry
            error,
        )  # error
        self.assertTrue(error.Success() and process, PROCESS_IS_VALID)

        # Create an empty event object.
        event = lldb.SBEvent()
        self.assertFalse(event, "Event should not be valid initially")

        # Create MyListeningThread to wait for any kind of event.
        import threading

        class MyListeningThread(threading.Thread):
            def run(self):
                count = 0
                # Let's only try at most 3 times to retrieve any kind of event.
                while not count > 3:
                    if listener.WaitForEvent(5, event):
                        self.context.trace("Got a valid event:", event)
                        self.context.trace("Event data flavor:", event.GetDataFlavor())
                        self.context.trace(
                            "Event type:", lldbutil.state_type_to_str(event.GetType())
                        )
                        listener.Clear()
                        return
                    count = count + 1
                    print("Timeout: listener.WaitForEvent")
                listener.Clear()
                return

        # Use Python API to kill the process.  The listening thread should be
        # able to receive a state changed event.
        process.Kill()

        # Let's start the listening thread to retrieve the event.
        my_thread = MyListeningThread()
        my_thread.context = self
        my_thread.start()

        # Wait until the 'MyListeningThread' terminates.
        my_thread.join()

        self.assertTrue(event, "My listening thread successfully received an event")

    @expectedFailureAll(
        oslist=["linux"], bugnumber="llvm.org/pr23617 Flaky, fails ~1/10 cases"
    )
    @skipIfWindows  # This is flakey on Windows AND when it fails, it hangs: llvm.org/pr38373
    @expectedFailureNetBSD
    def test_add_listener_to_broadcaster(self):
        """Exercise some SBBroadcaster APIs."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        self.dbg.SetAsync(True)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName("c", "a.out")
        self.trace("breakpoint:", breakpoint)
        self.assertTrue(
            breakpoint and breakpoint.GetNumLocations() == 1, VALID_BREAKPOINT
        )

        listener = lldb.SBListener("my listener")

        # Now launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        flags = target.GetLaunchInfo().GetLaunchFlags()
        process = target.Launch(
            listener,
            None,  # argv
            None,  # envp
            None,  # stdin_path
            None,  # stdout_path
            None,  # stderr_path
            None,  # working directory
            flags,  # launch flags
            False,  # Stop at entry
            error,
        )  # error

        # Create an empty event object.
        event = lldb.SBEvent()
        self.assertFalse(event, "Event should not be valid initially")

        # The finite state machine for our custom listening thread, with an
        # initial state of None, which means no event has been received.
        # It changes to 'connected' after 'connected' event is received (for remote platforms)
        # It changes to 'running' after 'running' event is received (should happen only if the
        # currentstate is either 'None' or 'connected')
        # It changes to 'stopped' if a 'stopped' event is received (should happen only if the
        # current state is 'running'.)
        self.state = None

        # Create MyListeningThread to wait for state changed events.
        # By design, a "running" event is expected following by a "stopped"
        # event.
        import threading

        class MyListeningThread(threading.Thread):
            def run(self):
                self.context.trace("Running MyListeningThread:", self)

                # Regular expression pattern for the event description.
                pattern = re.compile("data = {.*, state = (.*)}$")

                # Let's only try at most 6 times to retrieve our events.
                count = 0
                while True:
                    if listener.WaitForEvent(5, event):
                        desc = lldbutil.get_description(event)
                        self.context.trace("Event description:", desc)
                        match = pattern.search(desc)
                        if not match:
                            break
                        if match.group(1) == "connected":
                            # When debugging remote targets with lldb-server, we
                            # first get the 'connected' event.
                            self.context.assertTrue(self.context.state is None)
                            self.context.state = "connected"
                            continue
                        elif match.group(1) == "running":
                            self.context.assertTrue(
                                self.context.state is None
                                or self.context.state == "connected"
                            )
                            self.context.state = "running"
                            continue
                        elif match.group(1) == "stopped":
                            self.context.assertTrue(self.context.state == "running")
                            # Whoopee, both events have been received!
                            self.context.state = "stopped"
                            break
                        else:
                            break
                    print("Timeout: listener.WaitForEvent")
                    count = count + 1
                    if count > 6:
                        break
                listener.Clear()
                return

        # Use Python API to continue the process.  The listening thread should be
        # able to receive the state changed events.
        process.Continue()

        # Start the listening thread to receive the "running" followed by the
        # "stopped" events.
        my_thread = MyListeningThread()
        # Supply the enclosing context so that our listening thread can access
        # the 'state' variable.
        my_thread.context = self
        my_thread.start()

        # Wait until the 'MyListeningThread' terminates.
        my_thread.join()

        # The final judgement. :-)
        self.assertEqual(
            self.state, "stopped", "Both expected state changed events received"
        )

    def wait_for_next_event(self, expected_state, test_shadow=False):
        """Wait for an event from self.primary & self.shadow listener.
        If test_shadow is true, we also check that the shadow listener only
        receives events AFTER the primary listener does."""
        import stop_hook
        # Waiting on the shadow listener shouldn't have events yet because
        # we haven't fetched them for the primary listener yet:
        event = lldb.SBEvent()

        if test_shadow:
            success = self.shadow_listener.WaitForEvent(1, event)
            self.assertFalse(success, "Shadow listener doesn't pull events")

        # But there should be an event for the primary listener:
        success = self.primary_listener.WaitForEvent(5, event)

        self.assertTrue(success, "Primary listener got the event")

        state = lldb.SBProcess.GetStateFromEvent(event)
        primary_event_type = event.GetType()
        restart = False
        if state == lldb.eStateStopped:
            restart = lldb.SBProcess.GetRestartedFromEvent(event)
            # This counter is matching the stop hooks, which don't get run
            # for auto-restarting stops.
            if not restart:
                self.stop_counter += 1
                self.assertEqual(
                    stop_hook.StopHook.counter[self.instance],
                    self.stop_counter,
                    "matching stop hook",
                )

        if expected_state is not None:
            self.assertEqual(
                state, expected_state, "Primary thread got the correct event"
            )

        # And after pulling that one there should be an equivalent event for the shadow
        # listener:
        success = self.shadow_listener.WaitForEvent(5, event)
        self.assertTrue(success, "Shadow listener got event too")
        shadow_event_type = event.GetType()
        self.assertEqual(
            primary_event_type, shadow_event_type, "It was the same event type"
        )
        self.assertEqual(
            state, lldb.SBProcess.GetStateFromEvent(event), "It was the same state"
        )
        self.assertEqual(
            restart,
            lldb.SBProcess.GetRestartedFromEvent(event),
            "It was the same restarted",
        )
        return state, restart

    @expectedFlakeyLinux("llvm.org/pr23730")  # Flaky, fails ~1/100 cases
    @skipIfWindows  # This is flakey on Windows AND when it fails, it hangs: llvm.org/pr38373
    @skipIfNetBSD
    def test_shadow_listener(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        bkpt1 = target.BreakpointCreateByName("c", "a.out")
        self.trace("breakpoint:", bkpt1)
        self.assertEqual(bkpt1.GetNumLocations(), 1, VALID_BREAKPOINT)

        self.primary_listener = lldb.SBListener("my listener")
        self.shadow_listener = lldb.SBListener("shadow listener")

        self.cur_thread = None

        error = lldb.SBError()
        launch_info = target.GetLaunchInfo()
        launch_info.SetListener(self.primary_listener)
        launch_info.SetShadowListener(self.shadow_listener)

        self.runCmd(
            "settings set target.process.extra-startup-command QSetLogging:bitmask=LOG_PROCESS|LOG_EXCEPTIONS|LOG_RNB_PACKETS|LOG_STEP;"
        )
        self.dbg.SetAsync(True)

        # Now make our stop hook - we want to ensure it stays up to date with
        # the events.  We can't get our hands on the stop-hook instance directly,
        # so we'll pass in an instance key, and use that to retrieve the data from
        # this instance of the stop hook:
        self.instance = f"Key{random.randint(0,10000)}"
        stop_hook_path = os.path.join(self.getSourceDir(), "stop_hook.py")
        self.runCmd(f"command script import {stop_hook_path}")
        import stop_hook

        self.runCmd(
            f"target stop-hook add -P stop_hook.StopHook -k instance -v {self.instance}"
        )
        self.stop_counter = 0

        self.process = target.Launch(launch_info, error)
        self.assertSuccess(error, "Process launched successfully")

        # Keep fetching events from the primary to trigger the do on removal and
        # then from the shadow listener, and make sure they match:

        # Events in the launch sequence might be platform dependent, so don't
        # expect any particular event till we get the stopped:
        state = lldb.eStateInvalid

        while state != lldb.eStateStopped:
            state, restart = self.wait_for_next_event(None, False)

        # Okay, we're now at a good stop, so try a next:
        self.cur_thread = self.process.threads[0]

        # Make sure we're at our expected breakpoint:
        self.assertTrue(self.cur_thread.IsValid(), "Got a zeroth thread")
        self.assertEqual(self.cur_thread.stop_reason, lldb.eStopReasonBreakpoint)
        self.assertEqual(
            self.cur_thread.GetStopReasonDataCount(), 2, "Only one breakpoint/loc here"
        )
        self.assertEqual(
            bkpt1.GetID(),
            self.cur_thread.GetStopReasonDataAtIndex(0),
            "Hit the right breakpoint",
        )

        self.cur_thread.StepOver()
        # We'll run the test for "shadow listener blocked by primary listener
        # for the first couple rounds, then we'll skip the 1 second pause...
        self.wait_for_next_event(lldb.eStateRunning, True)
        self.wait_for_next_event(lldb.eStateStopped, True)

        # Next try an auto-continue breakpoint and make sure the shadow listener got
        # the resumed info as well.  Note that I'm not explicitly counting
        # running events here.  At the point when I wrote this lldb sometimes
        # emits two running events in a row.  Apparently the code to coalesce running
        # events isn't working.  But that's not what this test is testing, we're really
        # testing that the primary & shadow listeners hear the same thing and in the
        # right order.

        main_spec = lldb.SBFileSpec("main.c")
        bkpt2 = target.BreakpointCreateBySourceRegex("b.2. returns %d", main_spec)
        self.assertGreater(bkpt2.GetNumLocations(), 0, "BP2 worked")
        bkpt2.SetAutoContinue(True)

        bkpt3 = target.BreakpointCreateBySourceRegex("a.3. returns %d", main_spec)
        self.assertGreater(bkpt3.GetNumLocations(), 0, "BP3 worked")

        state = lldb.eStateStopped
        restarted = False

        # Put in a counter to make sure we don't spin forever if there is some
        # error in the logic.
        counter = 0
        while state != lldb.eStateExited:
            counter += 1
            self.assertLess(
                counter, 50, "Took more than 50 events to hit two breakpoints."
            )
            if state == lldb.eStateStopped and not restarted:
                self.process.Continue()

            state, restarted = self.wait_for_next_event(None, False)

        # Now make sure we agree with the stop hook counter:
        self.assertEqual(self.stop_counter, stop_hook.StopHook.counter[self.instance])
        self.assertEqual(stop_hook.StopHook.non_stops[self.instance], 0, "No non stops")
