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
        self.line_third_soft_trap = line_number("main.c", "// third soft trap:")
        self.line_fourth_soft_trap = line_number("main.c", "// fourth soft trap:")

    def _check_stop_reason_impl(
        self,
        expected_soft_trap_func: str,
        expected_stop_reason: str,
        expected_func_name: str,
        expected_file_name: str,
        expected_line_num: int,
        process=None,
    ):
        if process is None:
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
        self,
        stop_reason: str,
        func_name: str,
        file_name: str,
        line_num: int,
        process=None,
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
            process=process,
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

    def check_stopped_at_breakpoint(self, func_name: str, process=None):
        if process is None:
            process = self.test_target.process
        thread = process.GetSelectedThread()
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
        self.assertEqual(thread.GetSelectedFrame().GetFunctionName(), func_name)

    def bs_plugin_is_enabled(self, domain: str, debugger=None):
        return self.plugin_is_enabled(
            "instrumentation-runtime", "BoundsSafety", domain=domain, debugger=debugger
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
        self.build(make_targets=["soft-trap-test-minimal"])
        self.test_target = self.createTestTarget()

        try:
            # Disable the plugin so we do not stop at the first soft trap
            self.runCmd("plugin disable instrumentation-runtime.BoundsSafety")
            self.assertFalse(self.bs_plugin_is_enabled(domain="global"))

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

            # We should have skipped the first soft trap and stopped at the
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

    @skipIfWindows
    @skipUnlessBoundsSafety
    @no_debug_info_test
    def test_call_minimal_enable_disable_enable(self):
        """
        Test that the plugin can be toggled enable -> disable -> enable in a
        single session on code built with
        -fbounds-safety-soft-traps=call-minimal
        """
        self.build(make_targets=["soft-trap-test-minimal"])
        self.test_target = self.createTestTarget()
        self.runCmd("settings set target.env-vars DO_EXTRA_SOFT_TRAPS=1")

        bp1 = self.test_target.BreakpointCreateByName("test_breakpoint")
        self.assertTrue(bp1.GetNumLocations() > 0)
        bp2 = self.test_target.BreakpointCreateByName("test_breakpoint_2")
        self.assertTrue(bp2.GetNumLocations() > 0)

        self.runCmd("run")
        self.assertTrue(self.bs_plugin_is_enabled(domain="target"))
        self.assertTrue(self.bs_plugin_is_enabled(domain="global"))

        process = self.test_target.process

        # Step 1: ENABLED - we should hit the first soft trap.
        self.check_state_soft_trap_minimal(
            "Soft Bounds check failed: indexing above upper bound in 'buffer[2]'",
            "main",
            "main.c",
            self.line_first_soft_trap,
        )

        # Continue. Should hit test_breakpoint.
        process.Continue()
        self.check_stopped_at_breakpoint("test_breakpoint")

        # Step 2: DISABLE the plugin for this target.
        self.runCmd(
            "plugin disable --domain target instrumentation-runtime.BoundsSafety"
        )
        self.assertFalse(self.bs_plugin_is_enabled(domain="target"))
        self.assertTrue(self.bs_plugin_is_enabled(domain="global"))

        # Continue. Should skip the second and third soft traps and hit
        # test_breakpoint_2.
        process.Continue()
        self.check_stopped_at_breakpoint("test_breakpoint_2")

        # Step 3: RE-ENABLE the plugin for this target.
        self.runCmd(
            "plugin enable --domain target instrumentation-runtime.BoundsSafety"
        )
        self.assertTrue(self.bs_plugin_is_enabled(domain="target"))

        # Continue. Should hit the fourth soft trap.
        process.Continue()
        self.check_state_soft_trap_minimal(
            "Soft Bounds check failed: indexing below lower bound in 'buffer[-2]'",
            "main",
            "main.c",
            self.line_fourth_soft_trap,
        )

        # Continue. The process should exit cleanly.
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

    @skipIfWindows
    @skipUnlessBoundsSafety
    @no_debug_info_test
    def test_call_minimal_disable_enable_disable(self):
        """
        Test that the plugin can be toggled disable -> enable -> disable in a
        single session on code built with
        -fbounds-safety-soft-traps=call-minimal
        """
        self.build(make_targets=["soft-trap-test-minimal"])
        self.test_target = self.createTestTarget()
        self.runCmd("settings set target.env-vars DO_EXTRA_SOFT_TRAPS=1")

        try:
            # Disable the plugin globally before launching the process.
            self.runCmd("plugin disable instrumentation-runtime.BoundsSafety")
            self.assertFalse(self.bs_plugin_is_enabled(domain="global"))

            bp1 = self.test_target.BreakpointCreateByName("test_breakpoint")
            self.assertTrue(bp1.GetNumLocations() > 0)
            bp2 = self.test_target.BreakpointCreateByName("test_breakpoint_2")
            self.assertTrue(bp2.GetNumLocations() > 0)

            self.runCmd("run")

            process = self.test_target.process

            # Step 1: DISABLED - should skip the first soft trap and hit
            # test_breakpoint.
            self.check_stopped_at_breakpoint("test_breakpoint")
            self.assertFalse(self.bs_plugin_is_enabled(domain="target"))

            # Step 2: ENABLE the plugin for this target.
            self.runCmd(
                "plugin enable --domain target instrumentation-runtime.BoundsSafety"
            )
            self.assertTrue(self.bs_plugin_is_enabled(domain="target"))
            self.assertFalse(self.bs_plugin_is_enabled(domain="global"))

            # Continue. Should hit the second soft trap.
            process.Continue()
            self.check_state_soft_trap_minimal(
                "Soft Bounds check failed: indexing below lower bound in 'buffer[-1]'",
                "main",
                "main.c",
                self.line_second_soft_trap,
            )

            # Continue. Should hit the third soft trap.
            process.Continue()
            self.check_state_soft_trap_minimal(
                "Soft Bounds check failed: indexing above upper bound in 'buffer[3]'",
                "main",
                "main.c",
                self.line_third_soft_trap,
            )

            # Continue. Should hit test_breakpoint_2.
            process.Continue()
            self.check_stopped_at_breakpoint("test_breakpoint_2")

            # Step 3: DISABLE the plugin again.
            self.runCmd(
                "plugin disable --domain target instrumentation-runtime.BoundsSafety"
            )
            self.assertFalse(self.bs_plugin_is_enabled(domain="target"))

            # Continue. Should skip the fourth soft trap and exit cleanly.
            process.Continue()
            self.assertEqual(process.GetState(), lldb.eStateExited)
            self.assertEqual(process.GetExitStatus(), 0)
        finally:
            # Restore the global state to avoid affecting other tests
            self.runCmd("plugin enable instrumentation-runtime.BoundsSafety")
            self.assertTrue(self.bs_plugin_is_enabled(domain="global"))

    @skipIfWindows
    @skipUnlessBoundsSafety
    @skipIfRemote  # The dylib's directory compiled into a.out is a host path.
    @no_debug_info_test
    def test_call_minimal_disabled_plugin_stays_disabled_after_dlopen(self):
        """
        Test that a disabled plugin is not re-activated when a shared library
        containing the soft trap runtime functions is loaded via dlopen (which
        triggers ModulesDidLoad) on code built with
        -fbounds-safety-soft-traps=call-minimal
        """
        self.build(make_targets=["soft-trap-test-minimal"])
        self.test_target = self.createTestTarget()
        self.runCmd("settings set target.env-vars DO_DLOPEN=1")

        bp_dlopen = self.test_target.BreakpointCreateByName("test_breakpoint_dlopen")
        self.assertTrue(bp_dlopen.GetNumLocations() > 0)
        bp_after_dlopen = self.test_target.BreakpointCreateByName(
            "test_breakpoint_after_dlopen"
        )
        self.assertTrue(bp_after_dlopen.GetNumLocations() > 0)

        self.runCmd("run")
        self.assertTrue(self.bs_plugin_is_enabled(domain="target"))

        process = self.test_target.process

        # First soft trap hit
        self.check_state_soft_trap_minimal(
            "Soft Bounds check failed: indexing above upper bound in 'buffer[2]'",
            "main",
            "main.c",
            self.line_first_soft_trap,
        )

        # Disable the plugin for this target.
        self.runCmd(
            "plugin disable --domain target instrumentation-runtime.BoundsSafety"
        )
        self.assertFalse(self.bs_plugin_is_enabled(domain="target"))

        # Continue. The plugin is disabled so we should skip the second soft
        # trap and hit the breakpoint before dlopen.
        process.Continue()
        self.check_stopped_at_breakpoint("test_breakpoint_dlopen")
        self.assertFalse(self.bs_plugin_is_enabled(domain="target"))

        # Continue past dlopen. Loading a shared library that contains the soft
        # trap runtime functions triggers ModulesDidLoad. The plugin must stay
        # disabled and not re-activate. If the plugin re-activates, we would
        # stop on the soft trap inside the dylib instead of the breakpoint.
        process.Continue()
        self.check_stopped_at_breakpoint("test_breakpoint_after_dlopen")
        self.assertFalse(self.bs_plugin_is_enabled(domain="target"))

        # Make sure the dylib was actually loaded, otherwise this test isn't
        # testing anything.
        self.assertTrue(
            any(
                "dylib_with_soft_trap" in module.GetFileSpec().GetFilename()
                for module in self.test_target.module_iter()
            ),
            "dylib was not loaded",
        )

        # Continue. The process should exit cleanly.
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

    @skipIfWindows
    @skipUnlessBoundsSafety
    @skipIfRemote  # The second debugger isn't connected to the remote platform.
    @no_debug_info_test
    def test_call_minimal_global_domain_affects_all_debuggers(self):
        """
        Test that enabling/disabling the plugin in the global domain affects
        alive processes in all debuggers, not just the one running the command,
        on code built with -fbounds-safety-soft-traps=call-minimal
        """
        self.build(make_targets=["soft-trap-test-minimal"])
        self.test_target = self.createTestTarget()
        self.runCmd("settings set target.env-vars DO_EXTRA_SOFT_TRAPS=1")
        bp = self.test_target.BreakpointCreateByName("test_breakpoint_2")
        self.assertTrue(bp.GetNumLocations() > 0)
        exe = self.getBuildArtifact("a.out")

        other_dbg = lldb.SBDebugger.Create(False)
        other_dbg.SetAsync(False)
        other_process = None
        try:
            other_dbg.HandleCommand(
                "settings set target.env-vars DO_EXTRA_SOFT_TRAPS=1"
            )
            other_target = other_dbg.CreateTarget(exe)
            self.assertTrue(other_target, VALID_TARGET)
            other_bp = other_target.BreakpointCreateByName("test_breakpoint_2")
            self.assertTrue(other_bp.GetNumLocations() > 0)

            # Launch a process in each debugger. Both should stop on the first
            # soft trap.
            self.runCmd("run")
            process = self.test_target.process
            self.check_state_soft_trap_minimal(
                "Soft Bounds check failed: indexing above upper bound in 'buffer[2]'",
                "main",
                "main.c",
                self.line_first_soft_trap,
            )

            other_process = other_target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertTrue(other_process, PROCESS_IS_VALID)
            self.check_state_soft_trap_minimal(
                "Soft Bounds check failed: indexing above upper bound in 'buffer[2]'",
                "main",
                "main.c",
                self.line_first_soft_trap,
                process=other_process,
            )

            self.assertTrue(self.bs_plugin_is_enabled(domain="target"))
            self.assertTrue(
                self.bs_plugin_is_enabled(domain="target", debugger=other_dbg)
            )

            # Disable the plugin globally from the first debugger.
            self.runCmd(
                "plugin disable --domain global instrumentation-runtime.BoundsSafety"
            )
            self.assertFalse(self.bs_plugin_is_enabled(domain="global"))

            # The plugin should be disabled in the alive processes of both
            # debuggers.
            self.assertFalse(self.bs_plugin_is_enabled(domain="target"))
            self.assertFalse(
                self.bs_plugin_is_enabled(domain="target", debugger=other_dbg)
            )

            # Continue both processes. The second and third soft traps should
            # not cause instrumentation stops so both processes should stop at
            # test_breakpoint_2.
            process.Continue()
            self.check_stopped_at_breakpoint("test_breakpoint_2")
            other_process.Continue()
            self.check_stopped_at_breakpoint("test_breakpoint_2", process=other_process)

            # Enable the plugin globally from the first debugger. This creates
            # and activates the plugin in the alive processes of both debuggers.
            self.runCmd(
                "plugin enable --domain global instrumentation-runtime.BoundsSafety"
            )
            self.assertTrue(self.bs_plugin_is_enabled(domain="global"))
            self.assertTrue(self.bs_plugin_is_enabled(domain="target"))
            self.assertTrue(
                self.bs_plugin_is_enabled(domain="target", debugger=other_dbg)
            )

            # Continue both processes. Both should stop on the fourth soft trap.
            process.Continue()
            self.check_state_soft_trap_minimal(
                "Soft Bounds check failed: indexing below lower bound in 'buffer[-2]'",
                "main",
                "main.c",
                self.line_fourth_soft_trap,
            )
            other_process.Continue()
            self.check_state_soft_trap_minimal(
                "Soft Bounds check failed: indexing below lower bound in 'buffer[-2]'",
                "main",
                "main.c",
                self.line_fourth_soft_trap,
                process=other_process,
            )

            # Continue both processes. They should exit cleanly.
            process.Continue()
            self.assertEqual(process.GetState(), lldb.eStateExited)
            self.assertEqual(process.GetExitStatus(), 0)

            other_process.Continue()
            self.assertEqual(other_process.GetState(), lldb.eStateExited)
            self.assertEqual(other_process.GetExitStatus(), 0)
        finally:
            if other_process and other_process.IsValid():
                other_process.Kill()
            lldb.SBDebugger.Destroy(other_dbg)
            # Restore the global state to avoid affecting other tests.
            self.runCmd(
                "plugin enable --domain global instrumentation-runtime.BoundsSafety"
            )
            self.assertTrue(self.bs_plugin_is_enabled(domain="global"))

    @skipIfWindows
    @skipUnlessBoundsSafety
    @skipIfRemote  # The second debugger isn't connected to the remote platform.
    @no_debug_info_test
    def test_call_minimal_debugger_domain_only_affects_requesting_debugger(self):
        """
        Test that disabling the plugin in the debugger domain only affects
        alive processes in the debugger running the command on code built with
        -fbounds-safety-soft-traps=call-minimal
        """
        self.build(make_targets=["soft-trap-test-minimal"])
        self.test_target = self.createTestTarget()
        exe = self.getBuildArtifact("a.out")

        other_dbg = lldb.SBDebugger.Create(False)
        other_dbg.SetAsync(False)
        other_process = None
        try:
            other_target = other_dbg.CreateTarget(exe)
            self.assertTrue(other_target, VALID_TARGET)

            # Launch a process in each debugger. Both should stop on the first
            # soft trap.
            self.runCmd("run")
            process = self.test_target.process
            self.check_state_soft_trap_minimal(
                "Soft Bounds check failed: indexing above upper bound in 'buffer[2]'",
                "main",
                "main.c",
                self.line_first_soft_trap,
            )

            other_process = other_target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertTrue(other_process, PROCESS_IS_VALID)
            self.check_state_soft_trap_minimal(
                "Soft Bounds check failed: indexing above upper bound in 'buffer[2]'",
                "main",
                "main.c",
                self.line_first_soft_trap,
                process=other_process,
            )

            self.assertTrue(self.bs_plugin_is_enabled(domain="target"))
            self.assertTrue(
                self.bs_plugin_is_enabled(domain="target", debugger=other_dbg)
            )

            # Disable the plugin in the debugger domain of the first debugger.
            self.runCmd(
                "plugin disable --domain debugger instrumentation-runtime.BoundsSafety"
            )

            # The plugin should only be disabled in the first debugger. The
            # global enablement flag is not changed.
            self.assertFalse(self.bs_plugin_is_enabled(domain="target"))
            self.assertTrue(
                self.bs_plugin_is_enabled(domain="target", debugger=other_dbg)
            )
            self.assertTrue(self.bs_plugin_is_enabled(domain="global"))

            # Continue the first debugger's process. The second soft trap
            # should not cause an instrumentation stop and the process should
            # exit cleanly.
            process.Continue()
            self.assertEqual(process.GetState(), lldb.eStateExited)
            self.assertEqual(process.GetExitStatus(), 0)

            # Continue the other debugger's process. The plugin is still
            # enabled there so it should stop on the second soft trap.
            other_process.Continue()
            self.check_state_soft_trap_minimal(
                "Soft Bounds check failed: indexing below lower bound in 'buffer[-1]'",
                "main",
                "main.c",
                self.line_second_soft_trap,
                process=other_process,
            )
        finally:
            if other_process and other_process.IsValid():
                other_process.Kill()
            lldb.SBDebugger.Destroy(other_dbg)
