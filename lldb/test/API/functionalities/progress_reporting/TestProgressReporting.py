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

    def fetch_events(self):
        event = lldb.SBEvent()

        done = False
        while not done:
            if self.listener.WaitForEvent(1, event):
                event_mask = event.GetType();
                if event.BroadcasterMatchesRef(self.test_broadcaster):
                    if event_mask & self.eBroadcastBitStopProgressThread:
                        done = True;
                elif event.BroadcasterMatchesRef(self.progress_broadcaster):
                    ret_args = lldb.SBDebugger().GetProgressFromEvent(event);
                    self.assertGreater(len(ret_args), 1)

                    message = ret_args[0]
                    if message:
                        self.progress_events.append((message, event))

    def test_dwarf_symbol_loading_progress_report(self):
        """Test that we are able to fetch dwarf symbol loading progress events"""
        self.build()

        self.listener = lldb.SBListener("lldb.progress.listener")
        self.test_broadcaster = lldb.SBBroadcaster('lldb.broadcaster.test')
        self.listener.StartListeningForEvents(self.test_broadcaster,
                                              self.eBroadcastBitStopProgressThread)

        self.progress_broadcaster = self.dbg.GetBroadcaster()
        self.progress_broadcaster.AddListener(self.listener, lldb.SBDebugger.eBroadcastBitProgress)

        listener_thread = threading.Thread(target=self.fetch_events)
        listener_thread.start()

        lldbutil.run_to_source_breakpoint(self, 'break here', lldb.SBFileSpec('main.c'))

        self.test_broadcaster.BroadcastEventByType(self.eBroadcastBitStopProgressThread)
        listener_thread.join()

        self.assertGreater(len(self.progress_events), 0)
