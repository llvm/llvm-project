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

    def test_with_external_bit_set(self):
        """Test SBProgress can handle null events."""

        progress = lldb.SBProgress("Test SBProgress", "Test progress", 3, self.dbg)
        listener = lldb.SBListener("Test listener")
        broadcaster = self.dbg.GetBroadcaster()
        broadcaster.AddListener(listener, lldb.eBroadcastBitExternalProgress)
        event = lldb.SBEvent()
        # Sample JSON we're expecting:
        # { id = 2, title = "Test SBProgress", details = "Test progress", type = update, progress = 1 of 3}
        # details remains the same as specified in the constructor of the progress
        # until we update it in the increment function, so we check for the Null and empty string case
        # that details hasn't changed, but progress x of 3 has.
        progress.Increment(1, None)
        self.assertTrue(listener.GetNextEvent(event))
        stream = lldb.SBStream()
        event.GetDescription(stream)
        self.assertIn("Test progress", stream.GetData())
        self.assertIn("1 of 3", stream.GetData())

        progress.Increment(1, "")
        self.assertTrue(listener.GetNextEvent(event))
        event.GetDescription(stream)
        self.assertIn("Test progress", stream.GetData())
        self.assertIn("2 of 3", stream.GetData())

        progress.Increment(1, "Step 3")
        self.assertTrue(listener.GetNextEvent(event))
        stream = lldb.SBStream()
        event.GetDescription(stream)
        self.assertIn("Step 3", stream.GetData())
