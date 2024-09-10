"""
Test number of threads.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BreakpointAfterJoinTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number for our breakpoint.
        self.breakpoint = line_number("main.cpp", "// Set breakpoint here")

    @expectedFailureAll(
        oslist=["linux"],
        bugnumber="llvm.org/pr15824 thread states not properly maintained",
    )
    @expectedFailureAll(
        oslist=lldbplatformutil.getDarwinOSTriples(),
        bugnumber="llvm.org/pr15824 thread states not properly maintained and <rdar://problem/28557237>",
    )
    @expectedFailureAll(
        oslist=["freebsd"],
        bugnumber="llvm.org/pr18190 thread states not properly maintained",
    )
    @expectedFailureNetBSD
    def test(self):
        """Test breakpoint handling after a thread join."""
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint in the main thread.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.breakpoint, num_expected_locations=1
        )

        # The breakpoint list should show 1 location.
        self.expect(
            "breakpoint list -f",
            "Breakpoint location shown correctly",
            substrs=[
                "1: file = 'main.cpp', line = %d, exact_match = 0, locations = 1"
                % self.breakpoint
            ],
        )

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # Get the target process
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        # The exit probably occurred during breakpoint handling, but it isn't
        # guaranteed.  The main thing we're testing here is that the debugger
        # handles this cleanly is some way.

        # Get the number of threads
        num_threads = process.GetNumThreads()

        # Make sure we see at least six threads
        self.assertGreaterEqual(
            num_threads,
            6,
            "Number of expected threads and actual threads do not match.",
        )

        # Make sure all threads are stopped
        for i in range(0, num_threads):
            self.assertTrue(
                process.GetThreadAtIndex(i).IsStopped(),
                "Thread {0} didn't stop during breakpoint.".format(i),
            )

        # Run to completion
        self.runCmd("continue")

        # If the process hasn't exited, collect some information
        if process.GetState() != lldb.eStateExited:
            self.runCmd("thread list")
            self.runCmd("process status")

        # At this point, the inferior process should have exited.
        self.assertEqual(process.GetState(), lldb.eStateExited, PROCESS_EXITED)
