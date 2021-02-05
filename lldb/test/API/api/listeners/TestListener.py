"""
Test that we can listen to modules loaded events.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import six

class ListenToModuleLoadedEvents (TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def test_clearing_listener(self):
        """Make sure we also clear listerers from the hook up for event type manager"""
        self.build()

        my_first_listener = lldb.SBListener("bonus_listener")
        my_listener = lldb.SBListener("test_listener")
        my_third_listener = lldb.SBListener("extra_bonus_listener")

        my_listener.StartListeningForEventClass(
            self.dbg,
            lldb.SBTarget.GetBroadcasterClassName(),
            lldb.SBTarget.eBroadcastBitBreakpointChanged)
        my_first_listener.StartListeningForEventClass(
            self.dbg,
            lldb.SBTarget.GetBroadcasterClassName(),
            lldb.SBTarget.eBroadcastBitWatchpointChanged)
        my_third_listener.StartListeningForEventClass(
            self.dbg,
            lldb.SBTarget.GetBroadcasterClassName(),
            lldb.SBTarget.eBroadcastBitModulesUnloaded)

        exe = self.getBuildArtifact("a.out")

        my_listener.Clear()

        target = self.dbg.CreateTarget(exe)

        bkpt = target.BreakpointCreateByName("main")

        event = lldb.SBEvent()
        my_listener.WaitForEvent(1, event)
        self.assertTrue(not event.IsValid(), "We don't get events we aren't listening to.")
        
    def test_receiving_breakpoint_added_from_debugger(self):
        """Test that we get breakpoint added events, waiting on event classes on the debugger"""
        self.build()

        my_listener = lldb.SBListener("test_listener")

        my_listener.StartListeningForEventClass(
            self.dbg,
            lldb.SBTarget.GetBroadcasterClassName(),
            lldb.SBTarget.eBroadcastBitBreakpointChanged)

        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)

        bkpt = target.BreakpointCreateByName("main")

        event = lldb.SBEvent()
        my_listener.WaitForEvent(1, event)

        self.assertTrue(event.IsValid(), "Got a valid event.")
        self.assertTrue(
            lldb.SBBreakpoint.EventIsBreakpointEvent(event),
            "It is a breakpoint event.")
        self.assertTrue(lldb.SBBreakpoint.GetBreakpointEventTypeFromEvent(
            event) == lldb.eBreakpointEventTypeAdded, "It is a breakpoint added event.")
        self.assertEqual(
            bkpt, lldb.SBBreakpoint.GetBreakpointFromEvent(event),
            "It is our breakpoint.")

        # Now make sure if we stop listening for events we don't get them:
        my_listener.StopListeningForEventClass(
            self.dbg,
            lldb.SBTarget.GetBroadcasterClassName(),
            lldb.SBTarget.eBroadcastBitBreakpointChanged)
        my_listener.StopListeningForEvents(
            target.GetBroadcaster(),
            lldb.SBTarget.eBroadcastBitBreakpointChanged)
            
        bkpt2 = target.BreakpointCreateByName("main")
        my_listener.WaitForEvent(1, event)
        self.assertTrue(not event.IsValid(), "We don't get events we aren't listening to.")

    def test_recieving_breakpoint_added_from_target(self):
        """Test that we get breakpoint added events, waiting on event classes on the debugger"""
        self.build()

        my_listener = lldb.SBListener("test_listener")
        my_listener.StartListeningForEventClass(
            self.dbg,
            lldb.SBTarget.GetBroadcasterClassName(),
            lldb.SBTarget.eBroadcastBitBreakpointChanged)

        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        result = target.GetBroadcaster().AddListener(my_listener,
                                                     lldb.SBTarget.eBroadcastBitBreakpointChanged)
        self.assertEqual(result, lldb.SBTarget.eBroadcastBitBreakpointChanged,"Got our bit")

        bkpt = target.BreakpointCreateByName("main")

        event = lldb.SBEvent()
        my_listener.WaitForEvent(1, event)

        self.assertTrue(event.IsValid(), "Got a valid event.")
        self.assertTrue(
            lldb.SBBreakpoint.EventIsBreakpointEvent(event),
            "It is a breakpoint event.")
        self.assertTrue(lldb.SBBreakpoint.GetBreakpointEventTypeFromEvent(
            event) == lldb.eBreakpointEventTypeAdded, "It is a breakpoint added event.")
        self.assertEqual(
            bkpt, lldb.SBBreakpoint.GetBreakpointFromEvent(event),
            "It is our breakpoint.")

        # Now make sure if we stop listening for events we don't get them:
        target.GetBroadcaster().RemoveListener(my_listener)
            
        bkpt2 = target.BreakpointCreateByName("main")
        my_listener.WaitForEvent(1, event)
        self.assertTrue(not event.IsValid(), "We don't get events we aren't listening to.")
 
