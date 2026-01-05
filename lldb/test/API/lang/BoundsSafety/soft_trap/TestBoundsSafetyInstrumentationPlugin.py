"""
Test the BoundsSafety instrumentation plugin
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


STOP_REASON_MAX_LEN = 100
SOFT_TRAP_FUNC_MINIMAL = "__bounds_safety_soft_trap"
SOFT_TRAP_FUNC_WITH_STR = "__bounds_safety_soft_trap_s"


class BoundsSafetyTestSoftTrapPlugin(TestBase):
    def _check_stop_reason_impl(
        self,
        expected_soft_trap_func: str,
        expected_stop_reason: str,
        expected_func_name: str,
        expected_file_name: str,
        expected_line_num: int,
    ):
        process = self.test_target.process
        thread = process.GetSelectedThread()
        self.assertEqual(
            thread.GetStopReason(),
            lldb.eStopReasonInstrumentation,
        )

        stop_reason = thread.GetStopDescription(STOP_REASON_MAX_LEN)
        self.assertEqual(stop_reason, expected_stop_reason)

        soft_trap_func_frame = thread.GetFrameAtIndex(0)
        self.assertEqual(soft_trap_func_frame.name, expected_soft_trap_func)

        stop_frame = thread.GetSelectedFrame()
        self.assertEqual(stop_frame.name, expected_func_name)
        # The stop frame isn't frame 1 because that frame is the artificial
        # frame containing the trap reason.
        self.assertEqual(stop_frame.idx, 2)
        file_name = stop_frame.GetLineEntry().GetFileSpec().basename
        self.assertEqual(file_name, expected_file_name)
        line = stop_frame.GetLineEntry().line
        self.assertEqual(line, expected_line_num)

    def check_state_soft_trap_minimal(
        self, stop_reason: str, func_name: str, file_name: str, line_num: int
    ):
        """
        Check the program state is as expected when hitting
        a soft trap from -fbounds-safety-soft-traps=call-minimal
        """
        self._check_stop_reason_impl(
            SOFT_TRAP_FUNC_MINIMAL,
            expected_stop_reason=stop_reason,
            expected_func_name=func_name,
            expected_file_name=file_name,
            expected_line_num=line_num,
        )

    def check_state_soft_trap_with_str(
        self, stop_reason: str, func_name: str, file_name: str, line_num: int
    ):
        """
        Check the program state is as expected when hitting
        a soft trap from -fbounds-safety-soft-traps=call-with_str
        """
        self._check_stop_reason_impl(
            SOFT_TRAP_FUNC_WITH_STR,
            expected_stop_reason=stop_reason,
            expected_func_name=func_name,
            expected_file_name=file_name,
            expected_line_num=line_num,
        )

    # Skip the tests on Windows because they fail due to the stop reason
    # being `eStopReasonNon` instead of the expected
    # `eStopReasonInstrumentation`.
    @skipIfWindows
    @skipUnlessBoundsSafety
    def test_call_minimal(self):
        """
        Test the plugin on code built with
        -fbounds-safety-soft-traps=call-minimal
        """
        self.build(make_targets=["soft-trap-test-minimal"])
        self.test_target = self.createTestTarget()
        self.runCmd("run")

        process = self.test_target.process

        # First soft trap hit
        self.check_state_soft_trap_minimal(
            "Soft Bounds check failed: indexing above upper bound in 'buffer[2]'",
            "main",
            "main.c",
            7,
        )

        process.Continue()

        # Second soft trap hit
        self.check_state_soft_trap_minimal(
            "Soft Bounds check failed: indexing below lower bound in 'buffer[-1]'",
            "main",
            "main.c",
            8,
        )

        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

    @skipIfWindows
    @skipUnlessBoundsSafety
    def test_call_with_str(self):
        """
        Test the plugin on code built with
        -fbounds-safety-soft-traps=call-with-str
        """
        self.build(make_targets=["soft-trap-test-with-str"])
        self.test_target = self.createTestTarget()
        self.runCmd("run")

        process = self.test_target.process

        # First soft trap hit
        self.check_state_soft_trap_with_str(
            "Soft Bounds check failed: indexing above upper bound in 'buffer[2]'",
            "main",
            "main.c",
            7,
        )

        process.Continue()

        # Second soft trap hit
        self.check_state_soft_trap_with_str(
            "Soft Bounds check failed: indexing below lower bound in 'buffer[-1]'",
            "main",
            "main.c",
            8,
        )

        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)
