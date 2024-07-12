"""
Test lldb-dap setBreakpoints request
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase
from lldbsuite.test import lldbutil


class TestDAP_stopEvents(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    @skipIfRemote
    def test_single_stop_event(self):
        """
        Ensure single stopped event is sent during stop when singleStoppedEvent
        is set to True.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, singleStoppedEvent=True)
        source = "main.cpp"
        breakpoint_line = line_number(source, "// Set breakpoint1 here")
        lines = [breakpoint_line]
        # Set breakpoint in the thread function
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )
        self.continue_to_breakpoints(breakpoint_ids)
        self.assertEqual(
            len(self.dap_server.get_stopped_events()), 1, "expect one thread stopped"
        )

        loop_count = 10
        while loop_count > 0:
            self.dap_server.request_continue()
            stopped_event = self.dap_server.wait_for_stopped()
            self.assertEqual(
                len(self.dap_server.get_stopped_events()),
                1,
                "expect one thread stopped",
            )
            loop_count -= 1

    @skipIfWindows
    @skipIfRemote
    def test_correct_thread_count(self):
        """
        Test that the correct number of threads are reported in the stop event.
        No thread exited events are sent.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, singleStoppedEvent=True)
        source = "main.cpp"
        breakpoint_line = line_number(source, "// break worker thread here")
        lines = [breakpoint_line]
        # Set breakpoint in the thread function
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )
        self.continue_to_breakpoints(breakpoint_ids)

        threads = self.dap_server.get_threads()
        self.assertEqual(len(threads), 2, "expect two threads in first worker thread")

        self.dap_server.request_continue()
        stopped_event = self.dap_server.wait_for_stopped()
        threads = self.dap_server.get_threads()
        self.assertEqual(
            len(threads), 3, "expect three threads in second worker thread"
        )

        main_thread_breakpoint_line = line_number(source, "// break main thread here")
        # Set breakpoint in the thread function
        main_breakpoint_ids = self.set_source_breakpoints(
            source, [main_thread_breakpoint_line]
        )
        self.continue_to_breakpoints(main_breakpoint_ids)

        threads = self.dap_server.get_threads()
        self.assertEqual(
            len(threads), 3, "expect three threads in second worker thread"
        )

        exited_threads = self.dap_server.get_thread_events("exited")
        self.assertEqual(
            len(exited_threads),
            0,
            "expect no threads exited after hitting main thread breakpoint during context switch",
        )
