"""
End-to-end test for accelerator plugin actions (breakpoints and connections).

Launches a real process against an lldb-server that has the mock accelerator
plugin enabled and verifies that the breakpoints requested by the plugin are
set in the native process, hit, and that hitting one breakpoint can request
further breakpoints. This exercises all three breakpoint types: by name, by
name scoped to a shared library, and by address.

It also verifies that hitting the plugin's connection-trigger breakpoint causes
the client to create a second (accelerator) target and connect it to the mock
accelerator GDB server.
"""

import os

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


class MockAcceleratorActionsTestCase(TestBase):
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

    def set_mock_env(self, name, value):
        """Set an environment variable the mock plugin reads (it is inherited by
        the lldb-server that hosts the plugin), restoring it after the test."""
        previous = os.environ.get(name)
        os.environ[name] = value

        def restore():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

        self.addTearDownHook(restore)

    @skipIfRemote
    @add_test_categories(["llgs"])
    def test_accelerator_actions(self):
        """The mock accelerator plugin drives breakpoints in the inferior and,
        once initialized, a connection that creates a second target."""
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
        # plugin to request three more breakpoints: the connection hook (hit
        # next, since main() connects right after initializing), one by address
        # (on "mock_gpu_accelerator_compute", from the symbol value delivered
        # with the hit), and one by name scoped to the "a.out" shared library (on
        # "mock_gpu_accelerator_finish"). Only the native target exists until the
        # connection hook is hit.
        self.assertEqual(self.dbg.GetNumTargets(), 1)
        process.Continue()
        self.check_accelerator_breakpoint_stop(
            process, "mock_gpu_accelerator_connect", hit_count=1
        )

        # The accelerator target now exists alongside the native target.
        self.assertEqual(self.dbg.GetNumTargets(), 2)
        accelerator_target = None
        for i in range(self.dbg.GetNumTargets()):
            candidate = self.dbg.GetTargetAtIndex(i)
            if candidate != target:
                accelerator_target = candidate
                break
        self.assertTrue(accelerator_target.IsValid())

        # The accelerator process must be successfully connected and stopped.
        accelerator_process = accelerator_target.GetProcess()
        self.assertTrue(accelerator_process.IsValid())
        self.assertState(accelerator_process.GetState(), lldb.eStateStopped)

        # Validate the registers (each value == 0x1000 + register index).
        accelerator_frame = accelerator_process.GetThreadAtIndex(0).GetFrameAtIndex(0)
        expected_registers = {
            "r0": 0x1000,
            "r1": 0x1001,
            "sp": 0x1002,
            "fp": 0x1003,
            "pc": 0x1004,
            "flags": 0x1005,
        }
        for name, value in expected_registers.items():
            reg = accelerator_frame.FindRegister(name)
            self.assertTrue(reg.IsValid(), "register %s should exist" % name)
            self.assertEqual(
                reg.GetValueAsUnsigned(),
                value,
                "register %s should read back its expected value" % name,
            )

        # With the accelerator connected, continue through the remaining
        # breakpoint types: by address (on mock_gpu_accelerator_compute), then by
        # name scoped to a shared library (on mock_gpu_accelerator_finish).
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

    def run_expecting_failed_connection(self):
        """Run the inferior through its breakpoints, asserting the connection is
        attempted at the connection hook but no accelerator target is created
        (and lldb does not crash), then runs to exit."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        self.check_accelerator_breakpoint_stop(
            process, "mock_gpu_accelerator_initialize", hit_count=1
        )

        # At the connection hook the plugin returns connect_info the client
        # cannot use, so no second target is created and lldb does not crash.
        process.Continue()
        self.check_accelerator_breakpoint_stop(
            process, "mock_gpu_accelerator_connect", hit_count=1
        )
        self.assertEqual(
            self.dbg.GetNumTargets(), 1, "connection should fail; no accelerator target"
        )

        # The native process is unaffected: the remaining breakpoints still fire
        # and it runs to exit.
        process.Continue()
        self.check_accelerator_breakpoint_stop(
            process, "mock_gpu_accelerator_compute", hit_count=1
        )
        process.Continue()
        self.check_accelerator_breakpoint_stop(
            process, "mock_gpu_accelerator_finish", hit_count=1
        )
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

    @skipIfRemote
    @add_test_categories(["llgs"])
    def test_accelerator_connection_invalid_platform(self):
        """An invalid platform name in connect_info fails the connection
        gracefully."""
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_PLATFORM", "no-such-platform")
        self.run_expecting_failed_connection()

    @skipIfRemote
    @add_test_categories(["llgs"])
    def test_accelerator_connection_incompatible_triple(self):
        """A valid platform with a triple it does not support fails the
        connection gracefully."""
        # remote-linux is a real platform but cannot handle a GPU triple.
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_PLATFORM", "remote-linux")
        self.set_mock_env("LLDB_MOCK_ACCELERATOR_TRIPLE", "amdgcn-amd-amdhsa")
        self.run_expecting_failed_connection()
