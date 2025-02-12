import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


@skipIfAsan  # rdar://138777205
class TestCase(lldbtest.TestBase):

    def check_is_in_line(self, thread, expected_linenum, expected_tid):
        """Checks that thread has tid == expected_tid and is stopped at expected_linenum"""
        self.assertEqual(expected_tid, thread.GetThreadID())

        frame = thread.frames[0]
        line_entry = frame.GetLineEntry()
        self.assertEqual(expected_linenum, line_entry.GetLine())

    @swiftTest
    @skipIf(oslist=["windows", "linux"])
    def test_step_over_main(self):
        self.build()

        source_file = lldb.SBFileSpec("main.swift")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Breakpoint main", source_file
        )

        main_task_id = thread.GetThreadID()
        main_first_line = thread.frames[0].GetLineEntry().GetLine()
        num_lines_main = 7

        for line_offset in range(1, num_lines_main):
            thread.StepOver()
            self.check_is_in_line(thread, main_first_line + line_offset, main_task_id)

    @swiftTest
    @skipIf(oslist=["windows", "linux"])
    def test_step_over_top_level_fibonacci(self):
        self.build()

        source_file = lldb.SBFileSpec("main.swift")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Breakpoint main", source_file
        )

        main_task_id = thread.GetThreadID()
        fib_bp = target.BreakpointCreateBySourceRegex("Breakpoint fib", source_file)
        lldbutil.continue_to_breakpoint(process, fib_bp)

        # Get any of the threads that might have reached this breakpoint.
        # Any thread should work: their initial value for `n` is > 2,
        # so all threads do the full function body the first time around.
        thread_in_fib = lldbutil.get_threads_stopped_at_breakpoint(process, fib_bp)[0]
        thread_id_in_fib = thread_in_fib.GetThreadID()
        fib_bp.SetEnabled(False)

        fib_first_line = thread_in_fib.frames[0].GetLineEntry().GetLine()
        num_lines_fib = 5
        for line_offset in range(1, num_lines_fib):
            thread_in_fib.StepOver()
            self.assertEqual(process.GetSelectedThread(), thread_in_fib)
            self.check_is_in_line(
                thread_in_fib, fib_first_line + line_offset, thread_id_in_fib
            )
