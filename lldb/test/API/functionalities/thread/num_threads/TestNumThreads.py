"""
Test number of threads.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class NumberOfThreadsTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers for our break points.
        self.thread3_notify_all_line = line_number(
            "main.cpp", "// Set thread3 break point on notify_all at this line."
        )
        self.thread3_before_lock_line = line_number(
            "main.cpp", "// thread3-before-lock"
        )

    def test_number_of_threads(self):
        """Test number of threads."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint with 1 location.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.thread3_notify_all_line, num_expected_locations=1
        )

        # The breakpoint list should show 1 location.
        self.expect(
            "breakpoint list -f",
            "Breakpoint location shown correctly",
            substrs=[
                "1: file = 'main.cpp', line = %d, exact_match = 0, locations = 1"
                % self.thread3_notify_all_line
            ],
        )

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Stopped once.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint 1."],
        )

        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        # Get the number of threads
        num_threads = process.GetNumThreads()

        # Using std::thread may involve extra threads, so we assert that there are
        # at least 4 rather than exactly 4.
        self.assertTrue(
            num_threads >= 13,
            "Number of expected threads and actual threads do not match.",
        )

    @skipIfDarwin  # rdar://33462362
    @skipIfWindows  # This is flakey on Windows: llvm.org/pr37658, llvm.org/pr38373
    def test_unique_stacks(self):
        """Test backtrace unique with multiple threads executing the same stack."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Set a break point on the thread3 notify all (should get hit on threads 4-13).
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.thread3_before_lock_line, num_expected_locations=1
        )

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Stopped once.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint 1."],
        )

        process = self.process()

        # Get the number of threads
        num_threads = process.GetNumThreads()

        # Using std::thread may involve extra threads, so we assert that there are
        # at least 10 thread3's rather than exactly 10.
        self.assertTrue(
            num_threads >= 10,
            "Number of expected threads and actual threads do not match.",
        )

        # Attempt to walk each of the thread's executing the thread3 function to
        # the same breakpoint.
        def is_thread3(thread):
            for frame in thread:
                if frame.GetFunctionName() is None:
                    continue
                if "thread3" in frame.GetFunctionName():
                    return True
            return False

        expect_threads = ""
        for i in range(num_threads):
            thread = process.GetThreadAtIndex(i)
            self.assertTrue(thread.IsValid())
            if not is_thread3(thread):
                continue

            # If we aren't stopped out the thread breakpoint try to resume.
            if thread.GetStopReason() != lldb.eStopReasonBreakpoint:
                self.runCmd("thread continue %d" % (i + 1))
            self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)

            expect_threads += " #%d" % (i + 1)

        # Construct our expected back trace string
        expect_string = "10 thread(s)%s" % (expect_threads)

        # Now that we are stopped, we should have 10 threads waiting in the
        # thread3 function. All of these threads should show as one stack.
        self.expect(
            "thread backtrace unique",
            "Backtrace with unique stack shown correctly",
            substrs=[expect_string, "main.cpp:%d" % self.thread3_before_lock_line],
        )
