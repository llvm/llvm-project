"""
Test that we are able to broadcast and receive diagnostic events from lldb
"""
import lldb

import lldbsuite.test.lldbutil as lldbutil

from lldbsuite.test.lldbtest import *

class TestDiagnosticReporting(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

        self.broadcaster = self.dbg.GetBroadcaster()
        self.listener = lldbutil.start_listening_from(self.broadcaster,
                                        lldb.SBDebugger.eBroadcastBitWarning |
                                        lldb.SBDebugger.eBroadcastBitError)

    def test_dwarf_symbol_loading_diagnostic_report(self):
        """Test that we are able to fetch diagnostic events"""

        self.yaml2obj("minidump.yaml", self.getBuildArtifact("minidump.core"))

        self.dbg.CreateTarget(None)
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore(
            self.getBuildArtifact("minidump.core"))

        event = lldbutil.fetch_next_event(self, self.listener, self.broadcaster)
        diagnostic_data = lldb.SBDebugger.GetDiagnosticFromEvent(event)
        self.assertEquals(
            diagnostic_data.GetValueForKey("type").GetStringValue(100),
            "warning")
        self.assertEquals(
            diagnostic_data.GetValueForKey("message").GetStringValue(100),
            "unable to retrieve process ID from minidump file, setting process ID to 1"
        )
