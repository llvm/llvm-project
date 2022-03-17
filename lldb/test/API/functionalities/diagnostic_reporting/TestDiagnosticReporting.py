"""
Test that we are able to broadcast and receive diagnostic events from lldb
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import threading


class TestDiagnosticReporting(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    eBroadcastBitStopDiagnosticThread = (1 << 0)

    def setUp(self):
        TestBase.setUp(self)
        self.diagnostic_events = []

    def fetch_events(self):
        event = lldb.SBEvent()

        done = False
        while not done:
            if self.listener.WaitForEvent(1, event):
                event_mask = event.GetType()
                if event.BroadcasterMatchesRef(self.test_broadcaster):
                    if event_mask & self.eBroadcastBitStopDiagnosticThread:
                        done = True
                elif event.BroadcasterMatchesRef(self.diagnostic_broadcaster):
                    self.diagnostic_events.append(
                        lldb.SBDebugger.GetDiagnosticFromEvent(event))

    def test_dwarf_symbol_loading_diagnostic_report(self):
        """Test that we are able to fetch diagnostic events"""
        self.listener = lldb.SBListener("lldb.diagnostic.listener")
        self.test_broadcaster = lldb.SBBroadcaster('lldb.broadcaster.test')
        self.listener.StartListeningForEvents(
            self.test_broadcaster, self.eBroadcastBitStopDiagnosticThread)

        self.diagnostic_broadcaster = self.dbg.GetBroadcaster()
        self.diagnostic_broadcaster.AddListener(
            self.listener, lldb.SBDebugger.eBroadcastBitWarning)
        self.diagnostic_broadcaster.AddListener(
            self.listener, lldb.SBDebugger.eBroadcastBitError)

        listener_thread = threading.Thread(target=self.fetch_events)
        listener_thread.start()

        self.yaml2obj("minidump.yaml", self.getBuildArtifact("minidump.core"))

        self.dbg.CreateTarget(None)
        self.target = self.dbg.GetSelectedTarget()
        self.process = self.target.LoadCore(
            self.getBuildArtifact("minidump.core"))

        self.test_broadcaster.BroadcastEventByType(
            self.eBroadcastBitStopDiagnosticThread)
        listener_thread.join()

        self.assertEquals(len(self.diagnostic_events), 1)

        diagnostic_event = self.diagnostic_events[0]
        self.assertEquals(
            diagnostic_event.GetValueForKey("type").GetStringValue(100),
            "warning")
        self.assertEquals(
            diagnostic_event.GetValueForKey("message").GetStringValue(100),
            "unable to retrieve process ID from minidump file, setting process ID to 1"
        )
