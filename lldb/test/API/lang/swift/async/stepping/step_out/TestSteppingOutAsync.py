import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import re


class TestCase(lldbtest.TestBase):

    def check_and_get_frame_names(self, process):
        frames = process.GetSelectedThread().frames
        # Expected frames:
        # ASYNC___0___ <- ASYNC___1___ <- ASYNC___2___ <- ASYNC___3___ <- Main
        num_frames = 5
        func_names = [frame.GetFunctionName() for frame in frames[:num_frames]]
        for idx in range(num_frames - 1):
            self.assertIn(f"ASYNC___{idx}___", func_names[idx])
        self.assertIn("Main", func_names[num_frames - 1])
        return func_names

    def step_out_checks(self, thread, expected_func_names):
        # Keep stepping out, comparing the top frame's name with the expected name.
        for expected_func_name in expected_func_names:
            error = lldb.SBError()
            thread.StepOut(error)
            self.assertSuccess(error, "step out failed")
            stop_reason = thread.GetStopReason()
            self.assertStopReason(stop_reason, lldb.eStopReasonPlanComplete)
            self.assertEqual(thread.frames[0].GetFunctionName(), expected_func_name)

    @swiftTest
    @skipIf(oslist=["windows", "linux"])
    @skipIf("rdar://133849022")
    def test(self):
        """Test `frame variable` in async functions"""
        self.build()

        source_file = lldb.SBFileSpec("main.swift")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "BREAK HERE", source_file
        )

        func_names = self.check_and_get_frame_names(process)
        self.step_out_checks(thread, func_names[1:])
