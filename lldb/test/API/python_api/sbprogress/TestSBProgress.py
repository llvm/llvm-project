"""Test the SBProgress API."""

import lldb
from lldbsuite.test.lldbtest import *


class SBProgressTestCase(TestBase):
    def test_with_external_bit_set(self):
        """Test SBProgress events are listened to when the external bit is set."""

        progress = lldb.SBProgress("Test SBProgress", "Test progress", self.dbg)
        listener = lldb.SBListener("Test listener")
        broadcaster = self.dbg.GetBroadcaster()
        broadcaster.AddListener(listener, lldb.eBroadcastBitExternalProgress)
        event = lldb.SBEvent()

        expected_string = "Test progress first increment"
        progress.Increment(1, expected_string)
        self.assertTrue(listener.PeekAtNextEvent(event))
        stream = lldb.SBStream()
        event.GetDescription(stream)
        self.assertIn(expected_string, stream.GetData())

    def test_without_external_bit_set(self):
        """Test SBProgress events are not listened to on the internal progress bit."""

        progress = lldb.SBProgress("Test SBProgress", "Test progress", self.dbg)
        listener = lldb.SBListener("Test listener")
        broadcaster = self.dbg.GetBroadcaster()
        broadcaster.AddListener(listener, lldb.eBroadcastBitProgress)
        event = lldb.SBEvent()

        expected_string = "Test progress first increment"
        progress.Increment(1, expected_string)
        self.assertFalse(listener.PeekAtNextEvent(event))
