"""Test that adding, deleting and modifying watchpoints sends the appropriate events."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestWatchpointEvents(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers that we will step to in main:
        self.main_source = "main.c"

    @add_test_categories(["pyapi"])
    def test_with_python_api(self):
        """Test that adding, deleting and modifying watchpoints sends the appropriate events."""
        self.build()

        self.main_source_spec = lldb.SBFileSpec(self.main_source)

        target, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// Put a breakpoint here.", self.main_source_spec
        )

        frame = thread.GetFrameAtIndex(0)
        local_var = frame.FindVariable("local_var")
        self.assertTrue(local_var.IsValid())

        self.listener = lldb.SBListener("com.lldb.testsuite_listener")
        self.target_bcast = target.GetBroadcaster()
        self.target_bcast.AddListener(
            self.listener, lldb.SBTarget.eBroadcastBitWatchpointChanged
        )
        self.listener.StartListeningForEvents(
            self.target_bcast, lldb.SBTarget.eBroadcastBitWatchpointChanged
        )

        error = lldb.SBError()
        local_watch = local_var.Watch(True, False, True, error)
        if not error.Success():
            self.fail(
                "Failed to make watchpoint for local_var: %s" % (error.GetCString())
            )

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

        self.assertEqual(
            local_watch.GetCondition(),
            condition,
            'make sure watchpoint condition is "' + condition + '"',
        )

        target.DeleteWatchpoint(local_watch.GetID())
        self.GetWatchpointEvent(
            lldb.eWatchpointEventTypeDisabled, lldb.eWatchpointEventTypeRemoved
        )

        # Re-create it so that we can check DeleteAllWatchpoints
        local_watch = local_var.Watch(True, False, True, error)
        if not error.Success():
            self.fail(
                "Failed to make watchpoint for local_var: %s" % (error.GetCString())
            )
        self.GetWatchpointEvent(lldb.eWatchpointEventTypeAdded)
        target.DeleteAllWatchpoints()
        self.GetWatchpointEvent(
            lldb.eWatchpointEventTypeDisabled, lldb.eWatchpointEventTypeRemoved
        )

    def GetWatchpointEvent(self, *event_types):
        # We added a watchpoint so we should get a watchpoint added event.
        event = lldb.SBEvent()
        for event_type in event_types:
            success = self.listener.WaitForEvent(1, event)
            self.assertTrue(success, "Successfully got watchpoint event")
            self.assertTrue(
                lldb.SBWatchpoint.EventIsWatchpointEvent(event),
                "Event is a watchpoint event.",
            )
            found_type = lldb.SBWatchpoint.GetWatchpointEventTypeFromEvent(event)
            self.assertEqual(
                found_type,
                event_type,
                "Event is not correct type, expected: %d, found: %d"
                % (event_type, found_type),
            )
        # There shouldn't be another event waiting around:
        found_event = self.listener.PeekAtNextEventForBroadcasterWithType(
            self.target_bcast, lldb.SBTarget.eBroadcastBitWatchpointChanged, event
        )
        if found_event:
            print("Found an event I didn't expect: ", event.GetType())

        self.assertTrue(not found_event, f"Only expected {len(event_types)} events.")
