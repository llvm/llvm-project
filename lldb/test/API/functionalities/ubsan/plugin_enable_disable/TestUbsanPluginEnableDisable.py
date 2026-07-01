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
        self.line_third_ubsan_issue = line_number("main.c", "// third ubsan issue")
        self.line_fourth_ubsan_issue = line_number("main.c", "// fourth ubsan issue")

    def ubsan_plugin_is_enabled(self, domain: str):
        return self.plugin_is_enabled(
            "instrumentation-runtime", "UndefinedBehaviorSanitizer", domain=domain
        )

    def check_stopped_at_ubsan_issue(self, line_num):
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
        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()

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

        # Disable the UBSan plugin globally before launching the process so that
        # it isn't loaded when the process starts.
        self.runCmd("plugin disable instrumentation-runtime.UndefinedBehaviorSanitizer")
        try:
            self.assertFalse(self.ubsan_plugin_is_enabled(domain="global"))

            # Set a breakpoint on test_breakpoint which is called just before
            # the last UBSan issue.
            bp = target.BreakpointCreateByName("test_breakpoint")
            self.assertTrue(bp.GetNumLocations() > 0)

            self.runCmd("run")

            process = self.dbg.GetSelectedTarget().process
            thread = process.GetSelectedThread()
            frame = thread.GetSelectedFrame()

            # We should have skipped most UBSan issues and stopped at the
            # test_breakpoint function.
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
            thread = process.GetSelectedThread()
            frame = thread.GetSelectedFrame()

            # We should now hit the last UBSan issue.
            self.check_stopped_at_ubsan_issue(self.line_third_ubsan_issue)

            # Continue past the fourth UBSan issue (plugin is still enabled).
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

        # Disable the UBSan plugin globally before launching the process.
        self.runCmd("plugin disable instrumentation-runtime.UndefinedBehaviorSanitizer")
        try:
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
        # ModulesDidLoad. The plugin must stay disabled and not re-activate.
        # If the plugin re-activates, we would stop on the ubsan issue
        # inside the dylib instead of exiting cleanly.
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)
