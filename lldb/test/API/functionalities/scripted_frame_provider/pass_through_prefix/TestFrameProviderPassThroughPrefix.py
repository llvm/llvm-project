"""
Test that a frame provider can pass through all frames from its parent
StackFrameList while modifying function names.
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil


class FrameProviderPassThroughPrefixTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.source = "main.c"

    @expectedFailureAll(oslist=["linux"], archs=["arm$"])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_pass_through_with_prefix(self):
        """
        Test that a provider can read every frame from its parent list and
        return them with a prefixed function name.

        The call stack is main -> foo -> bar -> baz, with a breakpoint in
        baz. After registering the provider, every frame's function name
        should be prefixed with 'my_custom_'.
        """
        self.build()

        (target, process, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source)
        )

        # Verify the original backtrace: baz, bar, foo, main.
        expected_names = ["baz", "bar", "foo", "main"]
        for i, name in enumerate(expected_names):
            frame = thread.GetFrameAtIndex(i)
            self.assertEqual(
                frame.GetFunctionName(),
                name,
                f"Frame {i} should be '{name}' before provider",
            )

        original_frame_count = thread.GetNumFrames()
        self.assertGreaterEqual(
            original_frame_count, 4, "Should have at least 4 frames"
        )

        # Import and register the provider.
        script_path = os.path.join(self.getSourceDir(), "frame_provider.py")
        self.runCmd("command script import " + script_path)

        error = lldb.SBError()
        provider_id = target.RegisterScriptedFrameProvider(
            "frame_provider.PrefixPassThroughProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(
            error.Success(), f"Should register provider successfully: {error}"
        )
        self.assertNotEqual(provider_id, 0, "Provider ID should be non-zero")

        # Frame count should be unchanged since we're passing through.
        new_frame_count = thread.GetNumFrames()
        self.assertEqual(
            new_frame_count,
            original_frame_count,
            "Frame count should be unchanged (pass-through provider)",
        )

        # Every frame should now have the 'my_custom_' prefix exactly once.
        prefix = "my_custom_"
        for i, name in enumerate(expected_names):
            frame = thread.GetFrameAtIndex(i)
            expected = prefix + name
            self.assertEqual(
                frame.GetFunctionName(),
                expected,
                f"Frame {i} should be '{expected}' after provider",
            )

    @expectedFailureAll(oslist=["linux"], archs=["arm$"])
    def test_provider_receives_parent_frames(self):
        """
        Test that the provider's input_frames come from the parent
        StackFrameList, not from the provider's own output.

        The provider peeks at input_frames[0] when constructing frame 1.
        If that name already carries the 'my_custom_' prefix, the provider
        knows it was given its own output list and flags the error by
        inserting a 'danger_will_robinson_' prefix. The test asserts that
        'danger_will_robinson_' never
        appears, proving the provider received the bare parent list.
        """
        self.build()

        (target, process, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source)
        )

        # Import and register the validating provider.
        script_path = os.path.join(self.getSourceDir(), "frame_provider.py")
        self.runCmd("command script import " + script_path)

        error = lldb.SBError()
        provider_id = target.RegisterScriptedFrameProvider(
            "frame_provider.ValidatingPrefixProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(
            error.Success(), f"Should register provider successfully: {error}"
        )
        self.assertNotEqual(provider_id, 0, "Provider ID should be non-zero")

        # No frame should contain 'danger_will_robinson_' — that would mean the provider
        # was handed its own output list instead of the parent list.
        expected_names = ["baz", "bar", "foo", "main"]
        prefix = "my_custom_"
        for i, name in enumerate(expected_names):
            frame = thread.GetFrameAtIndex(i)
            actual = frame.GetFunctionName()
            self.assertFalse(
                actual.startswith("danger_will_robinson_"),
                f"Frame {i}: provider got its own output list "
                f"(expected bare parent frames, got '{actual}')",
            )
            expected = prefix + name
            self.assertEqual(
                actual,
                expected,
                f"Frame {i} should be '{expected}' after provider",
            )
