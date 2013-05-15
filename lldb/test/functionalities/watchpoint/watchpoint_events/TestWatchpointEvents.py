"""Test that adding, deleting and modifying watchpoints sends the appropriate events."""

import os, time
import unittest2
import lldb
import lldbutil
from lldbtest import *

class TestWatchpointEvents (TestBase):

    mydir = os.path.join("functionalities", "watchpoint", "watchpoint_events")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_with_dsym_and_python_api(self):
        """Test that adding, deleting and modifying watchpoints sends the appropriate events."""
        self.buildDsym()
        self.step_over_stepping()

    @python_api_test
    @dwarf_test
    def test_with_dwarf_and_python_api(self):
        """Test that adding, deleting and modifying watchpoints sends the appropriate events."""
        self.buildDwarf()
        self.step_over_stepping()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers that we will step to in main:
        self.main_source = "main.c"

    def GetWatchpointEvent (self, event_type):
        # We added a watchpoint so we should get a watchpoint added event.
        event = lldb.SBEvent()
        success = self.listener.WaitForEvent (1, event)
        self.assertTrue(success == True, "Successfully got watchpoint event")
        self.assertTrue (lldb.SBWatchpoint.EventIsWatchpointEvent(event), "Event is a watchpoint event.")
        found_type = lldb.SBWatchpoint.GetWatchpointEventTypeFromEvent (event)
        self.assertTrue (found_type == event_type, "Event is not correct type, expected: %d, found: %d"%(event_type, found_type))
        # There shouldn't be another event waiting around:
        found_event = self.listener.PeekAtNextEventForBroadcasterWithType (self.target_bcast, lldb.SBTarget.eBroadcastBitBreakpointChanged, event)
        if found_event:
            print "Found an event I didn't expect: ", event

        self.assertTrue (not found_event, "Only one event per change.")

    def step_over_stepping(self):
        """Use Python APIs to test stepping over and hitting breakpoints."""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.main_source_spec = lldb.SBFileSpec (self.main_source)

        break_in_main = target.BreakpointCreateBySourceRegex ('// Put a breakpoint here.', self.main_source_spec)
        self.assertTrue(break_in_main, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint (process, break_in_main)

        if len(threads) != 1:
            self.fail ("Failed to stop at first breakpoint in main.")

        thread = threads[0]
        frame = thread.GetFrameAtIndex(0)
        local_var = frame.FindVariable ("local_var")
        self.assertTrue (local_var.IsValid())

        self.listener = lldb.SBListener("com.lldb.testsuite_listener")
        self.target_bcast = target.GetBroadcaster()
        self.target_bcast.AddListener (self.listener, lldb.SBTarget.eBroadcastBitWatchpointChanged)
        self.listener.StartListeningForEvents (self.target_bcast, lldb.SBTarget.eBroadcastBitWatchpointChanged)

        error = lldb.SBError()
        local_watch = local_var.Watch(True, True, True, error)
        if not error.Success():
            self.fail ("Failed to make watchpoint for local_var: %s"%(error.GetCString()))

        self.GetWatchpointEvent (lldb.eWatchpointEventTypeAdded)
        # Now change some of the features of this watchpoint and make sure we get events:
        local_watch.SetEnabled(False)
        self.GetWatchpointEvent (lldb.eWatchpointEventTypeDisabled)

        local_watch.SetIgnoreCount(10)
        self.GetWatchpointEvent (lldb.eWatchpointEventTypeIgnoreChanged)

        local_watch.SetCondition ("1 == 2")
        self.GetWatchpointEvent (lldb.eWatchpointEventTypeConditionChanged)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
