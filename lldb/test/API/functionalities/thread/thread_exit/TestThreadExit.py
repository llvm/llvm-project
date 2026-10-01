"""
Test number of threads.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


@requireThreadSupport
class ThreadExitTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers for our breakpoints.
        self.break_1 = line_number("main.cpp", "// Set first breakpoint here")
        self.break_2 = line_number("main.cpp", "// Set second breakpoint here")
        self.break_3 = line_number("main.cpp", "// Set third breakpoint here")
        self.break_4 = line_number("main.cpp", "// Set fourth breakpoint here")

    @staticmethod
    def thread_ids(threads):
        return {t.GetThreadID() for t in threads}

    @skipIfWindows  # This is flakey on Windows: llvm.org/pr38373
    def test(self):
        """Test thread exit handling."""
        if self.getArchitecture().lower() == "arm":
            # We require a separate debug info file to be able to backtrace starting
            # from a libc function. This file is provided by libc6-dbg on Linux.
            self.runCmd("settings set symbols.enable-external-lookup true")

        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint with 1 location.
        bp1_id = lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.break_1, num_expected_locations=1
        )
        bp2_id = lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.break_2, num_expected_locations=1
        )
        bp3_id = lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.break_3, num_expected_locations=1
        )
        bp4_id = lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.break_4, num_expected_locations=1
        )

        # The breakpoint list should show 1 locations.
        self.expect(
            "breakpoint list -f",
            "Breakpoint location shown correctly",
            substrs=[
                "1: file = 'main.cpp', line = %d, exact_match = 0, locations = 1"
                % self.break_1,
                "2: file = 'main.cpp', line = %d, exact_match = 0, locations = 1"
                % self.break_2,
                "3: file = 'main.cpp', line = %d, exact_match = 0, locations = 1"
                % self.break_3,
                "4: file = 'main.cpp', line = %d, exact_match = 0, locations = 1"
                % self.break_4,
            ],
        )

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)
        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        stopped_thread = lldbutil.get_one_thread_stopped_at_breakpoint_id(
            process, bp1_id
        )
        self.assertIsNotNone(stopped_thread, "Process is not stopped at breakpoint 1")

        bp1_tids = self.thread_ids(lldbutil.get_threads_in_executable(process))
        num_threads = len(bp1_tids)
        self.assertGreaterEqual(
            num_threads,
            2,
            "Number of expected threads and actual threads do not match at breakpoint 1.",
        )

        # Run to the second breakpoint
        self.runCmd("continue")
        stopped_thread = lldbutil.get_one_thread_stopped_at_breakpoint_id(
            process, bp2_id
        )
        self.assertIsNotNone(stopped_thread, "Process is not stopped at breakpoint 2")
        thread1_tid = stopped_thread.GetThreadID()

        # Update the number of threads
        bp2_tids = self.thread_ids(lldbutil.get_threads_in_executable(process))
        new_num_threads = len(bp2_tids)
        self.assertEqual(
            new_num_threads,
            num_threads + 1,
            "Number of expected threads did not increase by 1 at bp 2.",
        )
        (thread2_tid,) = bp2_tids - bp1_tids

        # Run to the third breakpoint
        self.runCmd("continue")
        stopped_thread = lldbutil.get_one_thread_stopped_at_breakpoint_id(
            process, bp3_id
        )
        self.assertIsNotNone(stopped_thread, "Process is not stopped at breakpoint 3")

        # thread1 was joined. Check it is gone from the thread list, not just
        # from the count: a stale entry must not have an a.out frame.
        self.assertNotIn(
            thread1_tid,
            self.thread_ids(process),
            "Exited thread1 is still listed at bp 3.",
        )

        # Update the number of threads
        new_num_threads = len(lldbutil.get_threads_in_executable(process))
        self.assertEqual(
            new_num_threads,
            num_threads,
            "Number of expected threads is not equal to original number of threads at bp 3.",
        )

        # Run to the fourth breakpoint
        self.runCmd("continue")
        stopped_thread = lldbutil.get_one_thread_stopped_at_breakpoint_id(
            process, bp4_id
        )
        self.assertIsNotNone(stopped_thread, "Process is not stopped at breakpoint 4")

        self.assertNotIn(
            thread2_tid,
            self.thread_ids(process),
            "Exited thread2 is still listed at bp 4.",
        )

        # Update the number of threads
        new_num_threads = len(lldbutil.get_threads_in_executable(process))
        self.assertEqual(
            new_num_threads,
            num_threads - 1,
            "Number of expected threads did not decrease by 1 at bp 4.",
        )

        # Run to completion
        self.runCmd("continue")

        # At this point, the inferior process should have exited.
        self.assertState(process.GetState(), lldb.eStateExited, PROCESS_EXITED)
