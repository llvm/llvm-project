"""
Test that we are able to broadcast and receive progress events from lldb
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import threading

class TestProgressReporting(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    eBroadcastBitStopProgressThread = (1 << 0)

    def setUp(self):
        TestBase.setUp(self)
        self.progress_events = []

    def fetch_events(self, test_broadcaster):
        listener = lldb.SBListener("lldb.progress.listener")
        listener.StartListeningForEvents(test_broadcaster,
                                         self.eBroadcastBitStopProgressThread)

        progress_broadcaster = self.dbg.GetBroadcaster()
        progress_broadcaster.AddListener(listener, lldb.SBDebugger.eBroadcastBitProgress)

        event = lldb.SBEvent()

        done = False
        while not done:
            if listener.WaitForEvent(1, event):
                event_mask = event.GetType();
                if event.BroadcasterMatchesRef(test_broadcaster):
                    if event_mask & self.eBroadcastBitStopProgressThread:
                        done = True;
                elif event.BroadcasterMatchesRef(progress_broadcaster):
                    message = lldb.SBDebugger().GetProgressFromEvent(event, 0, 0, 0, False);
                    if message:
                        self.progress_events.append((message, event))

    @skipUnlessDarwin
    def test_dwarf_symbol_loading_progress_report(self):
        """Test that we are able to fetch dwarf symbol loading progress events"""
        self.build()

        test_broadcaster = lldb.SBBroadcaster('lldb.broadcaster.test')
        listener_thread = threading.Thread(target=self.fetch_events,
                                           args=[test_broadcaster])
        listener_thread.start()

        lldbutil.run_to_source_breakpoint(self, 'break here', lldb.SBFileSpec('main.c'))

        test_broadcaster.BroadcastEventByType(self.eBroadcastBitStopProgressThread)
        listener_thread.join()

        self.assertTrue(len(self.progress_events) > 0)
