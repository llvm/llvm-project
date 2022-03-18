"""
Test that we are able to broadcast and receive progress events from lldb
"""
import lldb

import lldbsuite.test.lldbutil as lldbutil

from lldbsuite.test.lldbtest import *
from lldbsuite.test.eventlistener import EventListenerTestBase


class TestProgressReporting(EventListenerTestBase):

    mydir = TestBase.compute_mydir(__file__)
    event_mask = lldb.SBDebugger.eBroadcastBitProgress
    event_data_extractor = lldb.SBDebugger.GetProgressFromEvent

    def test_dwarf_symbol_loading_progress_report(self):
        """Test that we are able to fetch dwarf symbol loading progress events"""
        self.build()

        lldbutil.run_to_source_breakpoint(self, 'break here', lldb.SBFileSpec('main.c'))
        self.assertGreater(len(self.events), 0)
