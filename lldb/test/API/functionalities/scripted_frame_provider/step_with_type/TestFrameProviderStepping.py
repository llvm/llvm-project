"""
Test the "step with type" feature of scripted frames.
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil


class TestFrameProviderStepping(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # The frame list IDs used by 'bt --provider' match the descriptor IDs
    # returned by RegisterScriptedFrameProvider:
    # 0 = base unwinder, 1 = first provider, 2 = second provider, etc.
    UNWINDER_FRAME_LIST_ID = 0
    FIRST_PROVIDER_FRAME_LIST_ID = 1
    SECOND_PROVIDER_FRAME_LIST_ID = 2

    def setUp(self):
        TestBase.setUp(self)
        self.source = lldb.SBFileSpec("main.c")

    @expectedFailureAll(
        oslist=["linux"],
        archs=["arm$"],
        bugnumber="github.com/llvm/llvm-project/issues/191859",
    )
    def test_step_with_errors_api(self):
        """Test that errors in creating the step plan are reported"""
        self.do_test_with_errors(False)

    def test_step_with_errors_command(self):
        """Test that errors in creating the step plan are reported"""
        self.do_test_with_errors(True)

    def test_step_with_type_api(self):
        """
        Test that a provider can alter the meaning of step in a
        frame using the SB API's to step.
        """
        self.do_test(False)

    def test_step_with_type_command(self):
        """
        Test that a provider can alter the meaning of step in a
        frame using the command line commands to step.
        """
        self.do_test(True)

    def test_with_no_step(self):
        """Test that a returning an empty class name falls back to the
        standard stepping algorithms"""
        self.common_startup("frame_provider.NoStepProvider")
        # Just do this for the API:
        error = lldb.SBError()
        self.thread.StepOver(lldb.eOnlyDuringStepping, error)
        self.assertSuccess(error)
        # We should have only stepped once, so the counter should be
        # at 1
        self.assertEqual(self.g_counter.signed, 1, "We stepped once")

    def common_startup(self, provider_class):
        self.build()

        (target, _, self.thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Stop here to step", self.source
        )

        # Import and register the provider.
        script_path = os.path.join(self.getSourceDir(), "frame_provider.py")
        self.runCmd("command script import " + script_path)

        error = lldb.SBError()
        provider_id = target.RegisterScriptedFrameProvider(
            provider_class,
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(
            error.Success(), f"Should register provider successfully: {error}"
        )
        self.assertNotEqual(provider_id, 0, "Provider ID should be non-zero")
        # Now do some stepping and make sure each one goes twice:
        self.g_counter = target.FindFirstGlobalVariable("g_counter")
        self.assertSuccess(self.g_counter.error, "Got g_counter value")
        self.assertEqual(self.g_counter.signed, 0, "Starts at 0")

    def do_test_with_errors(self, use_command):
        self.common_startup("frame_provider.BadStepProvider")
        expected_error = "'frame_provider.Oops' that does not exist"

        if use_command:
            self.expect("thread step-over", substrs=[expected_error], error=True)
        else:
            step_error = lldb.SBError()
            self.thread.StepOver(lldb.eOnlyDuringStepping, step_error)
            self.assertTrue(step_error.fail, "Got a failure as expected")
            self.assertIn(expected_error, step_error.description, "Right error")

    def do_test(
        self,
        use_command,
    ):
        self.common_startup("frame_provider.CorrectStepProvider")

        if use_command:
            self.runCmd("thread step-over")
        else:
            step_error = lldb.SBError()
            self.thread.StepOver(lldb.eOnlyDuringStepping, step_error)
            self.assertSuccess(step_error)

        # We stepped over twice, so counter should be 2:
        self.assertEqual(self.g_counter.signed, 2, "We stepped twice")

        # Step in steps in twice, so we should be in bar:
        if use_command:
            self.runCmd("thread step-in")
        else:
            self.thread.StepInto()

        frame_0 = self.thread.frames[0]
        self.assertEqual(frame_0.name, "bar", "Stepped in twice")
        # We haven't run the increment the counter yet, so it's still 2:
        self.assertEqual(self.g_counter.signed, 2, "We stepped twice")

        # Now do a step out and make sure it goes back to main:
        if use_command:
            self.runCmd("thread step-out")
        else:
            self.thread.StepOut()

        frame_0 = self.thread.frames[0]
        self.assertEqual(frame_0.name, "main", "Stepped out twice")
        self.assertEqual(self.g_counter.signed, 3, "Step out twice updated counter")
