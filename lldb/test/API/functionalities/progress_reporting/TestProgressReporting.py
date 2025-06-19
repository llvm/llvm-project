"""
Test that we are able to broadcast and receive progress events from lldb
"""
import lldb
import threading

import lldbsuite.test.lldbutil as lldbutil

from lldbsuite.test.lldbtest import *


class TestProgressReporting(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.broadcaster = self.dbg.GetBroadcaster()
        self.listener = lldbutil.start_listening_from(
            self.broadcaster, lldb.SBDebugger.eBroadcastBitProgress
        )

    def test_wait_attach_progress_reporting(self):
        """Test that progress reports for wait attaching work as intended."""
        self.build()
        target = self.dbg.CreateTarget(None)

        # Wait attach to a process, then check to see that a progress report was created
        # and that its message is correct for waiting to attach to a process.
        class AttachThread(threading.Thread):
            def __init__(self, target):
                threading.Thread.__init__(self)
                self.target = target

            def run(self):
                self.target.AttachToProcessWithName(
                    lldb.SBListener(), "a.out", True, lldb.SBError()
                )

        thread = AttachThread(target)
        thread.start()

        event = lldbutil.fetch_next_event(self, self.listener, self.broadcaster)
        progress_data = lldb.SBDebugger.GetProgressDataFromEvent(event)
        message = progress_data.GetValueForKey("message").GetStringValue(100)
        self.assertEqual(message, "Waiting to attach to process")

        # Interrupt the process attach to keep the test from stalling.
        target.process.SendAsyncInterrupt()

        thread.join()

    def test_dwarf_symbol_loading_progress_report(self):
        """Test that we are able to fetch dwarf symbol loading progress events"""
        self.build()

        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))

        event = lldbutil.fetch_next_event(self, self.listener, self.broadcaster)
        ret_args = lldb.SBDebugger.GetProgressFromEvent(event)
        self.assertGreater(len(ret_args), 0)
        message = ret_args[0]
        self.assertGreater(len(message), 0)

    def test_dwarf_symbol_loading_progress_report_structured_data(self):
        """Test that we are able to fetch dwarf symbol loading progress events
        using the structured data API"""
        self.build()

        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))

        event = lldbutil.fetch_next_event(self, self.listener, self.broadcaster)
        progress_data = lldb.SBDebugger.GetProgressDataFromEvent(event)
        message = progress_data.GetValueForKey("message").GetStringValue(100)
        self.assertGreater(len(message), 0)
        details = progress_data.GetValueForKey("details").GetStringValue(100)
        self.assertGreater(len(details), 0)
