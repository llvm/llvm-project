"""
Tests enabling and disabling the UndefinedBehaviorSanitizer instrumentation
runtime plugin during a debug session.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class UbsanPluginEnableDisableTestCase(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.line_first_ubsan_issue = line_number("main.c", "// first ubsan issue")
        self.line_second_ubsan_issue = line_number("main.c", "// second ubsan issue")
        self.line_third_ubsan_issue = line_number("main.c", "// third ubsan issue")
        self.line_fourth_ubsan_issue = line_number("main.c", "// fourth ubsan issue")

    def ubsan_plugin_is_enabled(self, domain: str, debugger=None):
        return self.plugin_is_enabled(
            "instrumentation-runtime",
            "UndefinedBehaviorSanitizer",
            domain=domain,
            debugger=debugger,
        )

    def check_stopped_at_ubsan_issue(self, line_num, process=None):
        if process is None:
            process = self.dbg.GetSelectedTarget().process
        thread = process.GetSelectedThread()
        stop_reason = thread.GetStopReason()
        self.assertStopReason(stop_reason, lldb.eStopReasonInstrumentation)
        self.assertIn("__ubsan_on_report", thread.GetFrameAtIndex(0).GetFunctionName())
        backtraces = thread.GetStopReasonExtendedBacktraces(
            lldb.eInstrumentationRuntimeTypeUndefinedBehaviorSanitizer
        )
        self.assertEqual(backtraces.GetSize(), 1)

        # Check that we stopped at the expected line somewhere in the stacktrace
        found = False
        for i in range(thread.GetNumFrames()):
            frame = thread.GetFrameAtIndex(i)
            if frame.GetLineEntry().GetFileSpec().GetFilename() == "main.c":
                if frame.GetLineEntry().GetLine() == line_num:
                    found = True
        self.assertTrue(found)

    def check_stopped_at_breakpoint(self, func_name: str, process=None):
        if process is None:
            process = self.dbg.GetSelectedTarget().process
        thread = process.GetSelectedThread()
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
        self.assertEqual(thread.GetSelectedFrame().GetFunctionName(), func_name)

    @skipUnlessUndefinedBehaviorSanitizer
    @no_debug_info_test
    def test_disable_plugin_after_hit(self):
        """Test that disabling the UBSan plugin mid-session prevents further
        instrumentation stops."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.registerSanitizerLibrariesWithTarget(target)

        self.runCmd("run")
        self.assertTrue(self.ubsan_plugin_is_enabled(domain="target"))
        self.assertTrue(self.ubsan_plugin_is_enabled(domain="global"))

        process = self.dbg.GetSelectedTarget().process

        # We should have stopped on the first UBSan issue.
        self.check_stopped_at_ubsan_issue(self.line_first_ubsan_issue)

        # Disable the UBSan plugin for this target
        self.runCmd(
            "plugin disable --domain target instrumentation-runtime.UndefinedBehaviorSanitizer"
        )
        self.assertFalse(self.ubsan_plugin_is_enabled(domain="target"))
        # Globally the plugin is still marked as enabled
        self.assertTrue(self.ubsan_plugin_is_enabled(domain="global"))

        # Continue. The remaining UBSan issues should not cause
        # instrumentation stops and the process should exit cleanly.
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

    @skipUnlessUndefinedBehaviorSanitizer
    @no_debug_info_test
    def test_enable_plugin_after_disable(self):
        """Test that disabling the UBSan plugin before launch prevents
        instrumentation stops, and re-enabling it mid-session restores them."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.registerSanitizerLibrariesWithTarget(target)

        try:
            # Disable the UBSan plugin globally before launching the process so
            # that it isn't loaded when the process starts.
            self.runCmd(
                "plugin disable instrumentation-runtime.UndefinedBehaviorSanitizer"
            )
            self.assertFalse(self.ubsan_plugin_is_enabled(domain="global"))

            # Set a breakpoint on test_breakpoint which is called just before
            # the third UBSan issue.
            bp = target.BreakpointCreateByName("test_breakpoint")
            self.assertTrue(bp.GetNumLocations() > 0)

            self.runCmd("run")

            process = self.dbg.GetSelectedTarget().process
            thread = process.GetSelectedThread()
            frame = thread.GetSelectedFrame()

            # We should have skipped the first two UBSan issues and stopped at
            # the test_breakpoint function.
            stop_reason = thread.GetStopReason()
            self.assertStopReason(stop_reason, lldb.eStopReasonBreakpoint)
            self.assertIn("test_breakpoint", frame.GetFunctionName())

            # Re-enable the UBSan plugin for the target
            self.runCmd(
                "plugin enable --domain target instrumentation-runtime.UndefinedBehaviorSanitizer"
            )
            self.assertTrue(self.ubsan_plugin_is_enabled(domain="target"))
            self.assertFalse(self.ubsan_plugin_is_enabled(domain="global"))

            # Continue
            process.Continue()

            # We should now hit the third UBSan issue.
            self.check_stopped_at_ubsan_issue(self.line_third_ubsan_issue)

            # Continue. The plugin is still enabled so we should hit the fourth
            # UBSan issue.
            process.Continue()
            self.check_stopped_at_ubsan_issue(self.line_fourth_ubsan_issue)

            # Continue. The process should exit cleanly.
            process.Continue()
            self.assertEqual(process.GetState(), lldb.eStateExited)
            self.assertEqual(process.GetExitStatus(), 0)
        finally:
            # Now restore the global state to avoid affecting other tests.
            self.runCmd(
                "plugin enable --domain global instrumentation-runtime.UndefinedBehaviorSanitizer"
            )
            self.assertTrue(self.ubsan_plugin_is_enabled(domain="global"))

    @skipUnlessUndefinedBehaviorSanitizer
    @no_debug_info_test
    def test_enable_disable_enable(self):
        """Test that the UBSan plugin can be toggled enable -> disable -> enable
        in a single session."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.registerSanitizerLibrariesWithTarget(target)

        # Set breakpoints on both breakpoint functions.
        bp1 = target.BreakpointCreateByName("test_breakpoint")
        self.assertTrue(bp1.GetNumLocations() > 0)
        bp2 = target.BreakpointCreateByName("test_breakpoint_2")
        self.assertTrue(bp2.GetNumLocations() > 0)

        self.runCmd("run")
        self.assertTrue(self.ubsan_plugin_is_enabled(domain="target"))
        self.assertTrue(self.ubsan_plugin_is_enabled(domain="global"))

        process = self.dbg.GetSelectedTarget().process

        # Step 1: ENABLED - we should hit the first UBSan issue.
        self.check_stopped_at_ubsan_issue(self.line_first_ubsan_issue)

        # Step 2: DISABLE the plugin for this target.
        self.runCmd(
            "plugin disable --domain target instrumentation-runtime.UndefinedBehaviorSanitizer"
        )
        self.assertFalse(self.ubsan_plugin_is_enabled(domain="target"))
        self.assertTrue(self.ubsan_plugin_is_enabled(domain="global"))

        # Continue. Should skip the second UBSan issue and hit the breakpoint.
        process.Continue()
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
        self.assertIn("test_breakpoint", frame.GetFunctionName())

        # Step 3: RE-ENABLE the plugin for this target.
        self.runCmd(
            "plugin enable --domain target instrumentation-runtime.UndefinedBehaviorSanitizer"
        )
        self.assertTrue(self.ubsan_plugin_is_enabled(domain="target"))

        # Continue. Should hit the third UBSan issue.
        process.Continue()
        self.check_stopped_at_ubsan_issue(self.line_third_ubsan_issue)

        # Continue. Should hit test_breakpoint_2.
        process.Continue()
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
        self.assertIn("test_breakpoint_2", frame.GetFunctionName())

        # Continue. Should hit the fourth UBSan issue.
        process.Continue()
        self.check_stopped_at_ubsan_issue(self.line_fourth_ubsan_issue)

        # Continue. The process should exit cleanly.
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

    @skipUnlessUndefinedBehaviorSanitizer
    @no_debug_info_test
    def test_disable_enable_disable(self):
        """Test that the UBSan plugin can be toggled disable -> enable -> disable
        in a single session."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.registerSanitizerLibrariesWithTarget(target)

        try:
            # Disable the UBSan plugin globally before launching the process.
            self.runCmd(
                "plugin disable instrumentation-runtime.UndefinedBehaviorSanitizer"
            )
            self.assertFalse(self.ubsan_plugin_is_enabled(domain="global"))

            # Set breakpoints on both breakpoint functions.
            bp1 = target.BreakpointCreateByName("test_breakpoint")
            self.assertTrue(bp1.GetNumLocations() > 0)
            bp2 = target.BreakpointCreateByName("test_breakpoint_2")
            self.assertTrue(bp2.GetNumLocations() > 0)

            self.runCmd("run")

            process = self.dbg.GetSelectedTarget().process
            thread = process.GetSelectedThread()
            frame = thread.GetSelectedFrame()

            # Step 1: DISABLED - should skip issues 1 and 2, hit the breakpoint.
            self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
            self.assertIn("test_breakpoint", frame.GetFunctionName())

            # Step 2: ENABLE the plugin for this target.
            self.runCmd(
                "plugin enable --domain target instrumentation-runtime.UndefinedBehaviorSanitizer"
            )
            self.assertTrue(self.ubsan_plugin_is_enabled(domain="target"))
            self.assertFalse(self.ubsan_plugin_is_enabled(domain="global"))

            # Continue. Should hit the third UBSan issue.
            process.Continue()
            self.check_stopped_at_ubsan_issue(self.line_third_ubsan_issue)

            # Continue. Should hit test_breakpoint_2.
            process.Continue()
            thread = process.GetSelectedThread()
            frame = thread.GetSelectedFrame()
            self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
            self.assertIn("test_breakpoint_2", frame.GetFunctionName())

            # Step 3: DISABLE the plugin again.
            self.runCmd(
                "plugin disable --domain target instrumentation-runtime.UndefinedBehaviorSanitizer"
            )
            self.assertFalse(self.ubsan_plugin_is_enabled(domain="target"))

            # Continue. Should skip the fourth UBSan issue and exit cleanly.
            process.Continue()
            self.assertEqual(process.GetState(), lldb.eStateExited)
            self.assertEqual(process.GetExitStatus(), 0)
        finally:
            # Restore the global state to avoid affecting other tests.
            self.runCmd(
                "plugin enable --domain global instrumentation-runtime.UndefinedBehaviorSanitizer"
            )
            self.assertTrue(self.ubsan_plugin_is_enabled(domain="global"))

    @skipUnlessUndefinedBehaviorSanitizer
    @skipIfRemote  # The dylib's directory compiled into a.out is a host path.
    @no_debug_info_test
    def test_disabled_plugin_stays_disabled_after_dlopen(self):
        """Test that a disabled UBSan plugin is not re-activated when a new
        shared library is loaded via dlopen (which triggers ModulesDidLoad)."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.registerSanitizerLibrariesWithTarget(target)

        # Set a breakpoint on test_breakpoint_dlopen which is called just
        # before the dlopen call.
        bp = target.BreakpointCreateByName("test_breakpoint_dlopen")
        self.assertTrue(bp.GetNumLocations() > 0)

        # Set a breakpoint on test_breakpoint_after_dlopen which is called
        # after calling into the dylib.
        bp_after_dlopen = target.BreakpointCreateByName("test_breakpoint_after_dlopen")
        self.assertTrue(bp_after_dlopen.GetNumLocations() > 0)

        # Enable the dlopen test path via environment variable.
        self.runCmd("settings set target.env-vars DO_DLOPEN=1")

        self.runCmd("run")
        self.assertTrue(self.ubsan_plugin_is_enabled(domain="target"))

        process = self.dbg.GetSelectedTarget().process

        # We should have stopped on the first UBSan issue.
        self.check_stopped_at_ubsan_issue(self.line_first_ubsan_issue)

        # Disable the UBSan plugin for this target.
        self.runCmd(
            "plugin disable --domain target instrumentation-runtime.UndefinedBehaviorSanitizer"
        )
        self.assertFalse(self.ubsan_plugin_is_enabled(domain="target"))

        # Continue. The plugin is disabled so we should skip remaining ubsan
        # issues and hit the breakpoint before dlopen.
        process.Continue()
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
        self.assertIn("test_breakpoint_dlopen", frame.GetFunctionName())

        # The plugin should still be disabled before dlopen.
        self.assertFalse(self.ubsan_plugin_is_enabled(domain="target"))

        # Continue past dlopen. Loading a new shared library triggers
        # ModulesDidLoad which must not re-create the disabled plugin.
        process.Continue()
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonBreakpoint)
        self.assertEqual(frame.GetFunctionName(), "test_breakpoint_after_dlopen")
        self.assertFalse(self.ubsan_plugin_is_enabled(domain="target"))

        # Make sure the dylib was actually loaded, otherwise this test isn't
        # testing anything.
        self.assertTrue(
            any(
                "dylib_with_ubsan_issue" in module.GetFileSpec().GetFilename()
                for module in target.module_iter()
            ),
            "dylib was not loaded",
        )

        # Continue. The process should exit cleanly.
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

    @skipUnlessUndefinedBehaviorSanitizer
    @no_debug_info_test
    def test_extended_backtraces_after_disable(self):
        """Test that asking for the extended backtraces of an instrumentation
        stop after the UBSan plugin has been disabled doesn't crash."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.registerSanitizerLibrariesWithTarget(target)

        self.runCmd("run")
        process = target.GetProcess()

        # We should have stopped on the first UBSan issue. This also checks that
        # the extended backtraces are available while the plugin is enabled.
        self.check_stopped_at_ubsan_issue(self.line_first_ubsan_issue)

        # Disable the UBSan plugin for this target. The thread is still stopped
        # with the UBSan stop reason but the plugin instance that provides the
        # extended backtraces no longer exists.
        self.runCmd(
            "plugin disable --domain target instrumentation-runtime.UndefinedBehaviorSanitizer"
        )
        self.assertFalse(self.ubsan_plugin_is_enabled(domain="target"))

        thread = process.GetSelectedThread()
        self.assertStopReason(thread.GetStopReason(), lldb.eStopReasonInstrumentation)
        backtraces = thread.GetStopReasonExtendedBacktraces(
            lldb.eInstrumentationRuntimeTypeUndefinedBehaviorSanitizer
        )
        self.assertEqual(backtraces.GetSize(), 0)

        # Continue. The process should exit cleanly.
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)

    @skipUnlessUndefinedBehaviorSanitizer
    @skipIfRemote  # The second debugger isn't connected to the remote platform.
    @no_debug_info_test
    def test_global_domain_affects_all_debuggers(self):
        """Test that enabling/disabling the UBSan plugin in the global domain
        affects alive processes in all debuggers, not just the one running the
        command."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.registerSanitizerLibrariesWithTarget(target)
        bp = target.BreakpointCreateByName("test_breakpoint_2")
        self.assertTrue(bp.GetNumLocations() > 0)

        other_dbg = lldb.SBDebugger.Create(False)
        other_dbg.SetAsync(False)
        other_process = None
        try:
            other_target = other_dbg.CreateTarget(exe)
            self.assertTrue(other_target, VALID_TARGET)
            self.registerSanitizerLibrariesWithTarget(other_target)
            other_bp = other_target.BreakpointCreateByName("test_breakpoint_2")
            self.assertTrue(other_bp.GetNumLocations() > 0)

            # Launch a process in each debugger. Both should stop on the first
            # UBSan issue.
            self.runCmd("run")
            self.check_stopped_at_ubsan_issue(self.line_first_ubsan_issue)
            process = target.GetProcess()

            other_process = other_target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertTrue(other_process, PROCESS_IS_VALID)
            self.check_stopped_at_ubsan_issue(
                self.line_first_ubsan_issue, process=other_process
            )

            self.assertTrue(self.ubsan_plugin_is_enabled(domain="target"))
            self.assertTrue(
                self.ubsan_plugin_is_enabled(domain="target", debugger=other_dbg)
            )

            # Disable the plugin globally from the first debugger.
            self.runCmd(
                "plugin disable --domain global instrumentation-runtime.UndefinedBehaviorSanitizer"
            )
            self.assertFalse(self.ubsan_plugin_is_enabled(domain="global"))

            # The plugin should be disabled in the alive processes of both
            # debuggers.
            self.assertFalse(self.ubsan_plugin_is_enabled(domain="target"))
            self.assertFalse(
                self.ubsan_plugin_is_enabled(domain="target", debugger=other_dbg)
            )

            # Continue both processes. The second and third UBSan issues should
            # not cause instrumentation stops so both processes should stop at
            # test_breakpoint_2.
            process.Continue()
            self.check_stopped_at_breakpoint("test_breakpoint_2")
            other_process.Continue()
            self.check_stopped_at_breakpoint("test_breakpoint_2", process=other_process)

            # Enable the plugin globally from the first debugger. This creates
            # and activates the plugin in the alive processes of both debuggers.
            self.runCmd(
                "plugin enable --domain global instrumentation-runtime.UndefinedBehaviorSanitizer"
            )
            self.assertTrue(self.ubsan_plugin_is_enabled(domain="global"))
            self.assertTrue(self.ubsan_plugin_is_enabled(domain="target"))
            self.assertTrue(
                self.ubsan_plugin_is_enabled(domain="target", debugger=other_dbg)
            )

            # Continue both processes. Both should stop on the fourth UBSan
            # issue.
            process.Continue()
            self.check_stopped_at_ubsan_issue(self.line_fourth_ubsan_issue)
            other_process.Continue()
            self.check_stopped_at_ubsan_issue(
                self.line_fourth_ubsan_issue, process=other_process
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
                "plugin enable --domain global instrumentation-runtime.UndefinedBehaviorSanitizer"
            )
            self.assertTrue(self.ubsan_plugin_is_enabled(domain="global"))

    @skipUnlessUndefinedBehaviorSanitizer
    @skipIfRemote  # The second debugger isn't connected to the remote platform.
    @no_debug_info_test
    def test_debugger_domain_only_affects_requesting_debugger(self):
        """Test that disabling the UBSan plugin in the debugger domain only
        affects alive processes in the debugger running the command."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.registerSanitizerLibrariesWithTarget(target)

        other_dbg = lldb.SBDebugger.Create(False)
        other_dbg.SetAsync(False)
        other_process = None
        try:
            other_target = other_dbg.CreateTarget(exe)
            self.assertTrue(other_target, VALID_TARGET)
            self.registerSanitizerLibrariesWithTarget(other_target)

            # Launch a process in each debugger. Both should stop on the first
            # UBSan issue.
            self.runCmd("run")
            self.check_stopped_at_ubsan_issue(self.line_first_ubsan_issue)
            process = target.GetProcess()

            other_process = other_target.LaunchSimple(
                None, None, self.get_process_working_directory()
            )
            self.assertTrue(other_process, PROCESS_IS_VALID)
            self.check_stopped_at_ubsan_issue(
                self.line_first_ubsan_issue, process=other_process
            )

            self.assertTrue(self.ubsan_plugin_is_enabled(domain="target"))
            self.assertTrue(
                self.ubsan_plugin_is_enabled(domain="target", debugger=other_dbg)
            )

            # Disable the plugin in the debugger domain of the first debugger.
            self.runCmd(
                "plugin disable --domain debugger instrumentation-runtime.UndefinedBehaviorSanitizer"
            )

            # The plugin should only be disabled in the first debugger. The
            # global enablement flag is not changed.
            self.assertFalse(self.ubsan_plugin_is_enabled(domain="target"))
            self.assertTrue(
                self.ubsan_plugin_is_enabled(domain="target", debugger=other_dbg)
            )
            self.assertTrue(self.ubsan_plugin_is_enabled(domain="global"))

            # Continue the first debugger's process. The remaining UBSan issues
            # should not cause instrumentation stops and the process should
            # exit cleanly.
            process.Continue()
            self.assertEqual(process.GetState(), lldb.eStateExited)
            self.assertEqual(process.GetExitStatus(), 0)

            # Continue the other debugger's process. The plugin is still
            # enabled there so it should stop on the next UBSan issue.
            other_process.Continue()
            self.check_stopped_at_ubsan_issue(
                self.line_second_ubsan_issue, process=other_process
            )
        finally:
            if other_process and other_process.IsValid():
                other_process.Kill()
            lldb.SBDebugger.Destroy(other_dbg)
