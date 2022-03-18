"""
Test that we are able to broadcast and receive diagnostic events from lldb
"""
import lldb

import lldbsuite.test.lldbutil as lldbutil

from lldbsuite.test.lldbtest import *
from lldbsuite.test.eventlistener import EventListenerTestBase

class TestDiagnosticReporting(EventListenerTestBase):

    mydir = TestBase.compute_mydir(__file__)
    event_mask = lldb.SBDebugger.eBroadcastBitWarning | lldb.SBDebugger.eBroadcastBitError
    event_data_extractor = lldb.SBDebugger.GetDiagnosticFromEvent

    def test_dwarf_symbol_loading_diagnostic_report(self):
        """Test that we are able to fetch diagnostic events"""

        self.yaml2obj("minidump.yaml", self.getBuildArtifact("minidump.core"))

        self.dbg.CreateTarget(None)
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore(
            self.getBuildArtifact("minidump.core"))

        self.assertEquals(len(self.events), 1)

        diagnostic_event = self.events[0]
        self.assertEquals(
            diagnostic_event.GetValueForKey("type").GetStringValue(100),
            "warning")
        self.assertEquals(
            diagnostic_event.GetValueForKey("message").GetStringValue(100),
            "unable to retrieve process ID from minidump file, setting process ID to 1"
        )
