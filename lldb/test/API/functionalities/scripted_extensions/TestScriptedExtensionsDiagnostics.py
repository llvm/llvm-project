"""
Verify that exceptions raised inside scripted extension affordance methods
(or missing required abstract methods) are surfaced to the user.

For entry points with no return-channel for errors
(`ScriptedProcess::CreateInstance`, `OperatingSystemPython` ctor,
`ScriptedThreadPlan::DidPush`, `BreakpointResolverScripted` ctor) the
diagnostic is broadcast via `Debugger::ReportError` and asserted on a
listener.

Entry points that already return an `llvm::Expected` / `Status`
(`ScriptedThread::Create`, `ScriptedFrameProvider::CreateInstance`,
`StopHookScripted::SetScriptCallback`) propagate the detailed error
through their return type; tests for those are tracked as follow-up.
"""

import os

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import expectedFailureAll
from lldbsuite.test.lldbtest import TestBase


class TestScriptedExtensionsDiagnostics(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.broadcaster = self.dbg.GetBroadcaster()
        self.listener = lldbutil.start_listening_from(
            self.broadcaster,
            lldb.SBDebugger.eBroadcastBitWarning | lldb.SBDebugger.eBroadcastBitError,
        )
        script_path = os.path.join(
            self.getSourceDir(), "malformed_scripted_extensions.py"
        )
        self.runCmd("command script import " + script_path)

    def assert_diagnostic(self, expected_substring):
        event = lldbutil.fetch_next_event(self, self.listener, self.broadcaster)
        data = lldb.SBDebugger.GetDiagnosticFromEvent(event)
        self.assertTrue(data.IsValid(), "event has diagnostic data")
        message = data.GetValueForKey("message").GetStringValue(4096)
        self.assertIn(expected_substring, message)

    def create_target(self):
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, "valid target")
        return target

    # ------------------------------------------------------------------
    # ScriptedProcess - reports via ScriptedProcess::CreateInstance
    # ------------------------------------------------------------------

    def test_scripted_process_missing_methods(self):
        """A ScriptedProcess missing abstract methods should emit a
        diagnostic naming the missing method."""
        target = self.create_target()
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetProcessPluginName("ScriptedProcess")
        launch_info.SetScriptedProcessClassName(
            "malformed_scripted_extensions.MissingMethodsScriptedProcess"
        )
        error = lldb.SBError()
        target.Launch(launch_info, error)
        self.assertTrue(error.Fail(), "launch should fail")
        self.assert_diagnostic("read_memory_at_address")

    # ------------------------------------------------------------------
    # BreakpointResolverScripted - reports via
    # BreakpointResolverScripted::CreateImplementationIfNeeded.
    # `m_error` is set but never surfaced to the user, so ReportError is
    # the only user-visible channel.
    # ------------------------------------------------------------------

    # TODO: malformed_scripted_extensions needs a class whose __init__
    # raises (the current ExceptionScriptedBreakpointResolver only raises
    # from __callback__, which goes through Dispatch/Status). Tracked as
    # follow-up.
    @expectedFailureAll(bugnumber="needs an init-raising malformed class")
    def test_scripted_breakpoint_resolver_init_failure(self):
        target = self.create_target()
        target.BreakpointCreateFromScript(
            "malformed_scripted_extensions.ExceptionScriptedBreakpointResolver",
            lldb.SBStructuredData(),
            lldb.SBFileSpecList(),
            lldb.SBFileSpecList(),
        )
        self.assert_diagnostic("ExceptionScriptedBreakpointResolver")

    # ------------------------------------------------------------------
    # ScriptedThreadPlan - reports via ScriptedThreadPlan::DidPush. The
    # plan stores the error in m_error_str but never surfaces it; the
    # diagnostic is the user-visible channel.
    # ------------------------------------------------------------------

    @expectedFailureAll(bugnumber="needs an init-raising malformed class")
    def test_scripted_thread_plan_init_failure(self):
        target = self.create_target()
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, "valid process")
        thread = process.GetSelectedThread()
        self.assertTrue(thread, "valid thread")
        thread.StepUsingScriptedThreadPlan(
            "malformed_scripted_extensions.ExceptionScriptedThreadPlan"
        )
        self.assert_diagnostic("ExceptionScriptedThreadPlan")

    # ------------------------------------------------------------------
    # The remaining entry points need a live inferior process to trigger
    # their creation paths, or have no plugin implementation yet.
    # ------------------------------------------------------------------

    @expectedFailureAll(
        bugnumber="OperatingSystemPython needs a live process to load the OS plugin"
    )
    def test_operating_system_missing_methods(self):
        self.assert_diagnostic("get_thread_info")

    @expectedFailureAll(bugnumber="ScriptedPlatform has no plugin implementation yet")
    def test_scripted_platform_missing_methods(self):
        self.assert_diagnostic("list_processes")
