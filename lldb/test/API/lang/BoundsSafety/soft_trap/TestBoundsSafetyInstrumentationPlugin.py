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
    SHARED_BUILD_TESTCASE = False

    def setUp(self):
        TestBase.setUp(self)
        self.line_first_soft_trap = line_number("main.c", "// first soft trap:")
        self.line_second_soft_trap = line_number("main.c", "// second soft trap:")

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

    def bs_plugin_is_enabled(self, domain: str):
        return self.plugin_is_enabled(
            "instrumentation-runtime", "BoundsSafety", domain=domain
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
        self.assertTrue(self.bs_plugin_is_enabled(domain="global"))
        self.assertTrue(self.bs_plugin_is_enabled(domain="target"))

        # First soft trap hit
        self.check_state_soft_trap_minimal(
            "Soft Bounds check failed: indexing above upper bound in 'buffer[2]'",
            "main",
            "main.c",
            self.line_first_soft_trap,
        )

        process.Continue()

        # Second soft trap hit
        self.check_state_soft_trap_minimal(
            "Soft Bounds check failed: indexing below lower bound in 'buffer[-1]'",
            "main",
            "main.c",
            self.line_second_soft_trap,
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
        self.assertTrue(self.bs_plugin_is_enabled(domain="global"))
        self.assertTrue(self.bs_plugin_is_enabled(domain="target"))

        # First soft trap hit
        self.check_state_soft_trap_with_str(
            "Soft Bounds check failed: indexing above upper bound in 'buffer[2]'",
            "main",
            "main.c",
            self.line_first_soft_trap,
        )

        process.Continue()

        # Second soft trap hit
        self.check_state_soft_trap_with_str(
            "Soft Bounds check failed: indexing below lower bound in 'buffer[-1]'",
            "main",
            "main.c",
            self.line_second_soft_trap,
        )

        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

    @skipIfWindows
    @skipUnlessBoundsSafety
    @no_debug_info_test
    def test_call_minimal_enable_then_disable_plugin(self):
        """
        Test starting with the plugin enabled on code built with
        -fbounds-safety-soft-traps=call-minimal and then later disable the
        plugin
        """
        self.build(make_targets=["soft-trap-test-minimal"])
        self.test_target = self.createTestTarget()

        # Check the plugin is enabled before we run
        self.assertTrue(self.bs_plugin_is_enabled(domain="global"))
        self.runCmd("run")

        process = self.test_target.process

        # First soft trap hit
        self.check_state_soft_trap_minimal(
            "Soft Bounds check failed: indexing above upper bound in 'buffer[2]'",
            "main",
            "main.c",
            self.line_first_soft_trap,
        )

        # Disable the plugin on the target so we do not stop at the second soft trap
        self.runCmd(
            "plugin disable --domain target instrumentation-runtime.BoundsSafety"
        )
        self.assertFalse(self.bs_plugin_is_enabled(domain="target"))
        self.assertTrue(self.bs_plugin_is_enabled(domain="global"))

        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

    @skipIfWindows
    @skipUnlessBoundsSafety
    @no_debug_info_test
    def test_call_minimal_disable_then_enable_plugin(self):
        """
        Test starting with the plugin disabled on code built with
        -fbounds-safety-soft-traps=call-minimal and then later enable the
        plugin
        """
        # Disable the plugin so we do not stop at the second soft trap
        self.runCmd("plugin disable instrumentation-runtime.BoundsSafety")
        self.assertFalse(self.bs_plugin_is_enabled(domain="global"))

        self.build(make_targets=["soft-trap-test-minimal"])
        self.test_target = self.createTestTarget()

        try:
            # Set a breakpoint on test_breakpoint which is called just before
            # the last soft trap
            bp = self.test_target.BreakpointCreateByName("test_breakpoint")
            self.assertTrue(bp.GetNumLocations() > 0)
            self.runCmd("run")
            self.assertFalse(self.bs_plugin_is_enabled(domain="global"))
            self.assertFalse(self.bs_plugin_is_enabled(domain="target"))

            process = self.test_target.process
            thread = process.GetSelectedThread()
            frame = thread.GetSelectedFrame()

            # We should have skipped all UBSan issues and stopped at the
            # test_breakpoint function.
            stop_reason = thread.GetStopReason()
            self.assertStopReason(stop_reason, lldb.eStopReasonBreakpoint)
            self.assertIn("test_breakpoint", frame.GetFunctionName())

            # Enable the plugin so we stop at the second soft trap
            self.runCmd(
                "plugin enable --domain target instrumentation-runtime.BoundsSafety"
            )
            self.assertTrue(self.bs_plugin_is_enabled(domain="target"))
            self.assertFalse(self.bs_plugin_is_enabled(domain="global"))
            process.Continue()

            # Second soft trap hit
            self.check_state_soft_trap_minimal(
                "Soft Bounds check failed: indexing below lower bound in 'buffer[-1]'",
                "main",
                "main.c",
                self.line_second_soft_trap,
            )

            process.Continue()
            self.assertEqual(process.GetState(), lldb.eStateExited)
            self.assertEqual(process.GetExitStatus(), 0)
        finally:
            # Restore the global state to avoid affecting other tests
            self.runCmd("plugin enable instrumentation-runtime.BoundsSafety")
            self.assertTrue(self.bs_plugin_is_enabled(domain="global"))
