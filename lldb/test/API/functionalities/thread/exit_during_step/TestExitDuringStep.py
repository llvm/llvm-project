"""
Test number of threads.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@requireThreadSupport
class ExitDuringStepTestCase(TestBase):
    @expectedFailureAll(
        oslist=["windows"],
        archs=["aarch64"],
        bugnumber="https://github.com/llvm/llvm-project/pull/228391",
    )
    # https://github.com/llvm/llvm-project/issues/217961
    @skipIf(archs=["arm$"], oslist=["linux"])
    def test(self):
        """Test thread exit during step handling."""
        self.build()
        self.exit_during_step_base(
            "thread step-inst -m all-threads", "stop reason = instruction step", True
        )

    @expectedFailureAll(
        oslist=["windows"],
        archs=["aarch64"],
        bugnumber="https://github.com/llvm/llvm-project/pull/228391",
    )
    # https://github.com/llvm/llvm-project/issues/217961
    @skipIf(archs=["arm$"], oslist=["linux"])
    def test_step_over(self):
        """Test thread exit during step-over handling."""
        self.build()
        self.exit_during_step_base(
            "thread step-over -m all-threads", "stop reason = step over", False
        )

    @expectedFailureAll(
        oslist=["windows"],
        archs=["aarch64"],
        bugnumber="https://github.com/llvm/llvm-project/pull/228391",
    )
    # https://github.com/llvm/llvm-project/issues/217961
    @skipIf(archs=["arm$"], oslist=["linux"])
    def test_step_in(self):
        """Test thread exit during step-in handling."""
        self.build()
        self.exit_during_step_base(
            "thread step-in -m all-threads", "stop reason = step in", False
        )

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break and continue.
        self.breakpoint = line_number("main.cpp", "// Set breakpoint here")
        self.continuepoint = line_number("main.cpp", "// Continue from here")

    def exit_during_step_base(self, step_cmd, step_stop_reason, by_instruction):
        """Test thread exit during step handling."""
        if self.getArchitecture().lower() == "arm":
            # We require a separate debug info file to be able to backtrace starting
            # from a libc function. This file is provided by libc6-dbg on Linux.
            self.runCmd("settings set symbols.enable-external-lookup true")

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # This should create a breakpoint in the main thread.
        self.bp_num = lldbutil.run_break_set_by_file_and_line(
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

        # Count only the threads running a.out code: the OS can add a thread
        # between two stops (see lldbutil.get_threads_in_executable).
        bp_tids = {t.GetThreadID() for t in lldbutil.get_threads_in_executable(process)}
        num_threads = len(bp_tids)
        # Make sure we see all three threads
        self.assertGreaterEqual(
            num_threads,
            3,
            "Number of expected threads and actual threads do not match.",
        )

        stepping_thread = lldbutil.get_one_thread_stopped_at_breakpoint_id(
            process, self.bp_num
        )
        self.assertIsNotNone(
            stepping_thread, "Could not find a thread stopped at the breakpoint"
        )

        current_line = self.breakpoint
        stepping_frame = stepping_thread.GetFrameAtIndex(0)
        self.assertEqual(
            current_line,
            stepping_frame.GetLineEntry().GetLine(),
            "Starting line for stepping doesn't match breakpoint line.",
        )

        # Keep stepping until we've reached our designated continue point
        while current_line != self.continuepoint:
            # Since we're using the command interpreter to issue the thread command
            # (on the selected thread) we need to ensure the selected thread is the
            # stepping thread.
            if stepping_thread != process.GetSelectedThread():
                process.SetSelectedThread(stepping_thread)

            self.runCmd(step_cmd)

            frame = stepping_thread.GetFrameAtIndex(0)

            current_line = frame.GetLineEntry().GetLine()

            if by_instruction and current_line == 0:
                continue

            self.assertGreaterEqual(
                current_line,
                self.breakpoint,
                "Stepped to unexpected line, " + str(current_line),
            )
            self.assertLessEqual(
                current_line,
                self.continuepoint,
                "Stepped to unexpected line, " + str(current_line),
            )

        self.runCmd("thread list")

        # Update the number of threads
        new_num_threads = len(lldbutil.get_threads_in_executable(process))

        # Check to see that we reduced the number of threads as expected
        self.assertEqual(
            new_num_threads,
            num_threads - 1,
            "Number of threads did not reduce by 1 after thread exit.",
        )
        # The exited thread must be gone from the thread list, not just from
        # the count: a stale entry must not have an a.out frame.
        gone = bp_tids - {t.GetThreadID() for t in process}
        self.assertEqual(len(gone), 1, "The exited thread is still listed.")

        self.expect(
            "thread list",
            "Process state is stopped due to step",
            substrs=["stopped", step_stop_reason],
        )

        # Run to completion
        self.runCmd("continue")

        # At this point, the inferior process should have exited.
        self.assertState(process.GetState(), lldb.eStateExited, PROCESS_EXITED)
