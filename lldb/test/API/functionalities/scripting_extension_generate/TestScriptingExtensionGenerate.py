"""
Meta-test: generate every kind of scripting extension template on the fly via
`scripting extension generate`, then actually register and drive each one
through a live lldb debug session, to catch regressions in the generator
itself (e.g. broken imports, wrong base-class name, missing __init__
chaining) rather than just checking that a Python file was produced.

The stub methods lldb calls into may legitimately report failure (e.g. an
`is_alive` stub that returns None can make `process launch` report an
error), so success/failure of the driving command is not asserted. What is
asserted is that lldb never surfaces an unhandled Python traceback while
running the generated code.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestScriptingExtensionGenerate(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def generate_extension(self, kind, generate_all=False):
        """Run `scripting extension generate` for `kind`, returning the path
        to the generated file. Fails the test if generation errors out."""
        output_path = self.getBuildArtifact(f"generated_{kind}.py")
        if os.path.exists(output_path):
            os.remove(output_path)

        all_flag = " -a" if generate_all else ""
        self.runCmd(
            f"scripting extension generate{all_flag} -n Test -o {output_path} {kind}"
        )

        self.assertTrue(
            os.path.exists(output_path),
            f"scripting extension generate did not create {output_path}",
        )
        return output_path

    def run_and_assert_no_traceback(self, cmd):
        """Run `cmd` and fail the test if lldb surfaced an unhandled Python
        traceback while executing it. The command itself may still report
        failure (e.g. a stub method returning None) without that being a
        generator bug."""
        self.runCmd(cmd, check=False)
        output = self.res.GetOutput() + self.res.GetError()
        self.assertNotIn(
            "Traceback (most recent call last)",
            output,
            f"command {cmd!r} raised a Python exception:\n{output}",
        )

    def test_generate_operating_system(self):
        """OperatingSystem: generate, import, and activate via the
        target.process.python-os-plugin-path setting."""
        self.build()
        path = self.generate_extension("OperatingSystem")
        self.run_and_assert_no_traceback(f"command script import {path}")

        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))

        self.run_and_assert_no_traceback(
            f"settings set target.process.python-os-plugin-path {path}"
        )
        self.run_and_assert_no_traceback("thread list")
        self.runCmd("settings clear target.process.python-os-plugin-path", check=False)

    def test_generate_scripted_platform(self):
        """ScriptedPlatform: generate, import, and select via `platform
        select`."""
        path = self.generate_extension("ScriptedPlatform")
        self.run_and_assert_no_traceback(f"command script import {path}")

        self.run_and_assert_no_traceback(
            "platform select scripted-platform -C "
            "generated_ScriptedPlatform.TestScriptedPlatform"
        )

    def test_generate_scripted_process(self):
        """ScriptedProcess: generate, import, and launch via `process launch
        -C`."""
        self.build()
        path = self.generate_extension("ScriptedProcess")
        self.run_and_assert_no_traceback(f"command script import {path}")

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        self.run_and_assert_no_traceback(
            "process launch -C generated_ScriptedProcess.TestScriptedProcess"
        )

    def test_generate_scripted_thread_plan(self):
        """ScriptedThreadPlan: generate, import, and drive via `thread
        step-scripted -C`."""
        self.build()
        path = self.generate_extension("ScriptedThreadPlan")
        self.run_and_assert_no_traceback(f"command script import {path}")

        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))

        self.run_and_assert_no_traceback(
            "thread step-scripted -C "
            "generated_ScriptedThreadPlan.TestScriptedThreadPlan"
        )

    def test_generate_scripted_breakpoint_resolver(self):
        """ScriptedBreakpointResolver: generate, import, and set via
        `breakpoint set -P`."""
        self.build()
        path = self.generate_extension("ScriptedBreakpointResolver")
        self.run_and_assert_no_traceback(f"command script import {path}")

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        self.run_and_assert_no_traceback(
            "breakpoint set -P "
            "generated_ScriptedBreakpointResolver.TestScriptedBreakpointResolver"
        )

    def test_generate_scripted_hook_for_stop_hook(self):
        """ScriptedHook drives `target stop-hook add -P` (the stop-hook is a
        subset of the general scripted hook interface)."""
        self.build()
        path = self.generate_extension("ScriptedHook")
        self.run_and_assert_no_traceback(f"command script import {path}")

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        self.run_and_assert_no_traceback(
            "target stop-hook add -P generated_ScriptedHook.TestScriptedHook"
        )

    def test_generate_scripted_frame_provider(self):
        """ScriptedFrameProvider: generate, import, and register via `target
        frame-provider register -C`. Uses -a to generate all methods since
        `get_description` is a required static abstract method."""
        self.build()
        path = self.generate_extension("ScriptedFrameProvider", generate_all=True)
        self.run_and_assert_no_traceback(f"command script import {path}")

        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))

        self.run_and_assert_no_traceback(
            "target frame-provider register -C "
            "generated_ScriptedFrameProvider.TestScriptedFrameProvider"
        )

    def test_generate_scripted_hook(self):
        """ScriptedHook: generate, import, and register via `target hook add
        -P`."""
        self.build()
        path = self.generate_extension("ScriptedHook")
        self.run_and_assert_no_traceback(f"command script import {path}")

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        self.run_and_assert_no_traceback(
            "target hook add -P generated_ScriptedHook.TestScriptedHook"
        )
