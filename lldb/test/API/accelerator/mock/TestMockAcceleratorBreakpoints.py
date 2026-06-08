"""
End-to-end test for accelerator plugin breakpoints.

Launches a real process against an lldb-server that has the mock accelerator
plugin enabled and verifies that the breakpoints requested by the plugin are
set in the native process, hit, and that hitting one breakpoint can request
further breakpoints. This exercises all three breakpoint types: by name, by
name scoped to a shared library, and by address.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import configuration


def uint64_to_int64(value):
    """Reinterpret an unsigned 64-bit value as a signed 64-bit integer."""
    if value >= (1 << 63):
        return value - (1 << 64)
    return value


class MockAcceleratorBreakpointsTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        super().setUp()
        if "mock-accelerator" not in configuration.enabled_plugins:
            self.skipTest("mock-accelerator plugin is not enabled")

    def check_accelerator_breakpoint_stop(self, process, function_name, hit_count=None):
        """Verify the process stopped at an internal accelerator breakpoint in
        the given function. If hit_count is not None, also verify the
        breakpoint's hit count. Returns the breakpoint."""
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = process.GetSelectedThread()

        # The stop must be due to a breakpoint, and the frame must be in the
        # expected function.
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(frame.GetFunctionName(), function_name)

        # The breakpoint id is carried in the stop reason data. Accelerator
        # breakpoints are internal, so they are not in the public breakpoint
        # list, but can still be looked up by id. The datum is an unsigned
        # 64-bit value holding the (signed) breakpoint id; internal ids are
        # negative.
        self.assertGreater(thread.GetStopReasonDataCount(), 0)
        bp_id = uint64_to_int64(thread.GetStopReasonDataAtIndex(0))
        bp = process.GetTarget().FindBreakpointByID(bp_id)
        self.assertTrue(bp.IsValid())
        self.assertTrue(bp.IsInternal(), "accelerator breakpoints are internal")

        if hit_count is not None:
            self.assertEqual(bp.GetHitCount(), hit_count)
        return bp

    @skipIfRemote
    @add_test_categories(["llgs"])
    def test_accelerator_breakpoints(self):
        """The mock accelerator plugin drives breakpoints in the inferior."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launching the process should stop at the
        # "mock_gpu_accelerator_initialize" breakpoint that the mock plugin
        # requested via jAcceleratorPluginInitialize (it requests the native
        # process not auto-resume). This is a breakpoint by name with no shared
        # library.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        self.check_accelerator_breakpoint_stop(
            process, "mock_gpu_accelerator_initialize", hit_count=1
        )

        # The accelerator breakpoint was set and hit, yet it is internal, so it
        # never appears in the public breakpoint list.
        self.assertEqual(target.GetNumBreakpoints(), 0)

        # Hitting the mock_gpu_accelerator_initialize breakpoint caused the
        # plugin to request two more breakpoints: one by address (on
        # "mock_gpu_accelerator_compute", from the symbol value delivered with
        # the hit) and one by name scoped to the "a.out" shared library (on
        # "mock_gpu_accelerator_finish"). main() calls
        # mock_gpu_accelerator_compute() first.
        process.Continue()
        self.check_accelerator_breakpoint_stop(
            process, "mock_gpu_accelerator_compute", hit_count=1
        )

        process.Continue()
        self.check_accelerator_breakpoint_stop(
            process, "mock_gpu_accelerator_finish", hit_count=1
        )

        # No more accelerator breakpoints; the process runs to exit.
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)
