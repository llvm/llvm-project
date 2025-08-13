import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


@skipIfAsan  # rdar://138777205
class TestCase(lldbtest.TestBase):

    def check_x_is_available(self, frame):
        x_var = frame.FindVariable("x")
        self.assertTrue(x_var.IsValid(), f"Failed to find x in {frame}")
        self.assertEqual(x_var.GetValueAsUnsigned(), 30)

    def check_is_in_line(self, thread, linenum):
        frame = thread.frames[0]
        line_entry = frame.GetLineEntry()
        self.assertEqual(linenum, line_entry.GetLine())

    @swiftTest
    @skipIf(oslist=["windows", "linux"])
    def test(self):
        """Test conditions for async step-over."""
        self.build()

        source_file = lldb.SBFileSpec("main.swift")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "BREAK HERE", source_file
        )
        bkpt.SetEnabled(False) # avoid hitting multiple locations in async breakpoints

        expected_line_nums = [4]  # print(x)
        expected_line_nums += [5, 6, 7, 8, 5, 6, 7, 8, 5]  # two runs over the loop
        expected_line_nums += [9, 10]  # if line + if block
        for expected_line_num in expected_line_nums:
            thread.StepOver()
            stop_reason = thread.GetStopReason()
            self.assertStopReason(stop_reason, lldb.eStopReasonPlanComplete)
            self.check_is_in_line(thread, expected_line_num)
            self.check_x_is_available(thread.frames[0])
