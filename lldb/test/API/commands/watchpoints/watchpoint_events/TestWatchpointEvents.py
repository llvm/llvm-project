"""Test that adding, deleting and modifying watchpoints sends the appropriate events."""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestWatchpointEvents (TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers that we will step to in main:
        self.main_source = "main.c"

    @add_test_categories(['pyapi'])
    def test_with_python_api(self):
        """Test that adding, deleting and modifying watchpoints sends the appropriate events."""
        self.build()
        target = self.createTestTarget()

        self.main_source_spec = lldb.SBFileSpec(self.main_source)

        break_in_main = target.BreakpointCreateBySourceRegex(
            '// Put a breakpoint here.', self.main_source_spec)
        self.assertTrue(break_in_main, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, break_in_main)

        if len(threads) != 1:
            self.fail("Failed to stop at first breakpoint in main.")

        thread = threads[0]
        frame = thread.GetFrameAtIndex(0)
        local_var = frame.FindVariable("local_var")
        self.assertTrue(local_var.IsValid())

        self.listener = lldb.SBListener("com.lldb.testsuite_listener")
        self.target_bcast = target.GetBroadcaster()
        self.target_bcast.AddListener(
            self.listener, lldb.SBTarget.eBroadcastBitWatchpointChanged)
        self.listener.StartListeningForEvents(
            self.target_bcast, lldb.SBTarget.eBroadcastBitWatchpointChanged)

        error = lldb.SBError()
        local_watch = local_var.Watch(True, False, True, error)
        if not error.Success():
            self.fail(
                "Failed to make watchpoint for local_var: %s" %
                (error.GetCString()))

        self.GetWatchpointEvent(lldb.eWatchpointEventTypeAdded)
        # Now change some of the features of this watchpoint and make sure we
        # get events:
        local_watch.SetEnabled(False)
        self.GetWatchpointEvent(lldb.eWatchpointEventTypeDisabled)

        local_watch.SetEnabled(True)
        self.GetWatchpointEvent(lldb.eWatchpointEventTypeEnabled)

        local_watch.SetIgnoreCount(10)
        self.GetWatchpointEvent(lldb.eWatchpointEventTypeIgnoreChanged)

        condition = "1 == 2"
        local_watch.SetCondition(condition)
        self.GetWatchpointEvent(lldb.eWatchpointEventTypeConditionChanged)

        self.assertEqual(local_watch.GetCondition(), condition,
                        'make sure watchpoint condition is "' + condition + '"')

    def GetWatchpointEvent(self, event_type):
        # We added a watchpoint so we should get a watchpoint added event.
        event = lldb.SBEvent()
        success = self.listener.WaitForEvent(1, event)
        self.assertTrue(success, "Successfully got watchpoint event")
        self.assertTrue(
            lldb.SBWatchpoint.EventIsWatchpointEvent(event),
            "Event is a watchpoint event.")
        found_type = lldb.SBWatchpoint.GetWatchpointEventTypeFromEvent(event)
        self.assertEqual(
            found_type, event_type,
            "Event is not correct type, expected: %d, found: %d" %
            (event_type,
             found_type))
        # There shouldn't be another event waiting around:
        found_event = self.listener.PeekAtNextEventForBroadcasterWithType(
            self.target_bcast, lldb.SBTarget.eBroadcastBitBreakpointChanged, event)
        if found_event:
            print("Found an event I didn't expect: ", event)

        self.assertTrue(not found_event, "Only one event per change.")
