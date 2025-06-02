"""
Test lldb-dap threads request
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase


class TestDAP_threads(lldbdap_testcase.DAPTestCaseBase):
    def test_correct_thread(self):
        """
        Tests that the correct thread is selected if we continue from
        a thread that goes away and hit a breakpoint in another thread.
        In this case, the selected thread should be the thread that
        just hit the breakpoint, and not the first thread in the list.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        breakpoint_line = line_number(source, "// break here")
        lines = [breakpoint_line]
        # Set breakpoint in the thread function
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )
        self.continue_to_breakpoints(breakpoint_ids)
        # We're now stopped at the breakpoint in the first thread, thread #2.
        # Continue to join the first thread and hit the breakpoint in the
        # second thread, thread #3.
        self.dap_server.request_continue()
        stopped_event = self.dap_server.wait_for_stopped()
        # Verify that the description is the relevant breakpoint,
        # preserveFocusHint is False and threadCausedFocus is True
        self.assertTrue(
            stopped_event[0]["body"]["description"].startswith(
                "breakpoint %s." % breakpoint_ids[0]
            )
        )
        self.assertFalse(stopped_event[0]["body"]["preserveFocusHint"])
        self.assertTrue(stopped_event[0]["body"]["threadCausedFocus"])
        # All threads should be named Thread {index}
        threads = self.dap_server.get_threads()
        self.assertTrue(all(len(t["name"]) > 0 for t in threads))

    def test_thread_format(self):
        """
        Tests the support for custom thread formats.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(
            program,
            customThreadFormat="This is thread index #${thread.index}",
            stopCommands=["thread list"],
        )
        source = "main.cpp"
        breakpoint_line = line_number(source, "// break here")
        lines = [breakpoint_line]
        # Set breakpoint in the thread function
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )
        self.continue_to_breakpoints(breakpoint_ids)
        # We are stopped at the first thread
        threads = self.dap_server.get_threads()
        print("got thread", threads)
        if self.getPlatform() == "windows":
            # Windows creates a thread pool once WaitForSingleObject is called
            # by thread.join(). As we are in the thread function, we can't be
            # certain that join() has been called yet and a thread pool has
            # been created, thus we only check for the first two threads.
            names = list(sorted(t["name"] for t in threads))[:2]
            self.assertEqual(
                names, ["This is thread index #1", "This is thread index #2"]
            )
        else:
            self.assertEqual(threads[0]["name"], "This is thread index #1")
            self.assertEqual(threads[1]["name"], "This is thread index #2")
