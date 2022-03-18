"""
Test that we are able to broadcast and receive swift progress events from lldb
"""
import lldb

import lldbsuite.test.lldbutil as lldbutil

from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test.eventlistener import EventListenerTestBase

class TestSwiftProgressReporting(EventListenerTestBase):

    mydir = TestBase.compute_mydir(__file__)
    event_mask = lldb.SBDebugger.eBroadcastBitProgress
    event_data_extractor = lldb.SBDebugger.GetProgressFromEvent

    @swiftTest
    @skipIf(oslist=no_match(["macosx"]))
    def test_swift_progress_report(self):
        """Test that we are able to fetch swift type-system progress events"""
        self.build()

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(self, 'break here',
                                          lldb.SBFileSpec('main.swift'))

        self.assertGreater(thread.GetNumFrames(), 0)
        frame = thread.GetSelectedFrame()
        self.assertTrue(frame, "Invalid frame.")

        # Resolve variable to exercise the type-system
        self.runCmd("expr boo")

        self.assertGreater(len(self.events), 0)

        beacons = [ "Loading Swift module",
                    "Caching Swift user imports from",
                    "Setting up Swift reflection for",
                    "Getting Swift compile unit imports for"]

        for beacon in beacons:
            filtered_events = list(filter(lambda event: beacon in event[0],
                                          self.events))
            self.assertGreater(len(filtered_events), 0)
