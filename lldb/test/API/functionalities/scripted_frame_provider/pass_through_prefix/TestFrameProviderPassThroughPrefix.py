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

    # The frame list IDs used by 'bt --provider' are internal sequential IDs:
    # 0 = base unwinder, 1 = first provider, 2 = second provider, etc.
    # These are NOT the descriptor IDs returned by RegisterScriptedFrameProvider.
    UNWINDER_FRAME_LIST_ID = 0
    FIRST_PROVIDER_FRAME_LIST_ID = 1
    SECOND_PROVIDER_FRAME_LIST_ID = 2

    def setUp(self):
        TestBase.setUp(self)
        self.source = "main.c"

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

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_bt_provider_shows_unwinder_frames(self):
        """
        Test that 'bt --provider 0' shows the base unwinder frames
        (without the provider prefix) even after a provider is registered.
        """
        self.build()

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source)
        )

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

        # 'bt --provider 0' should show the base unwinder frames without prefix.
        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("bt --provider 0", result)
        self.assertTrue(result.Succeeded(), "bt --provider 0 should succeed")
        output = result.GetOutput()
        self.assertIn("Base Unwinder", output)
        self.assertIn("baz", output)
        self.assertNotIn("my_custom_", output)

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_bt_provider_shows_provider_frames(self):
        """
        Test that 'bt --provider <id>' shows the provider's transformed frames
        with the prefix applied.
        """
        self.build()

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source)
        )

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

        # 'bt --provider <id>' should show the provider's prefixed frames.
        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            f"bt --provider {self.FIRST_PROVIDER_FRAME_LIST_ID}", result
        )
        self.assertTrue(
            result.Succeeded(),
            f"bt --provider {self.FIRST_PROVIDER_FRAME_LIST_ID} should succeed",
        )
        output = result.GetOutput()
        self.assertIn("PrefixPassThroughProvider", output)
        self.assertIn("my_custom_baz", output)
        self.assertIn("my_custom_bar", output)
        self.assertIn("my_custom_foo", output)
        self.assertIn("my_custom_main", output)

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_bt_provider_range(self):
        """
        Test that 'bt --provider 0-<id>' shows both the base unwinder
        and provider frames sequentially.
        """
        self.build()

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source)
        )

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

        # 'bt --provider 0-<id>' should show both sections.
        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            f"bt --provider 0-{self.FIRST_PROVIDER_FRAME_LIST_ID}", result
        )
        self.assertTrue(
            result.Succeeded(),
            f"bt --provider 0-{self.FIRST_PROVIDER_FRAME_LIST_ID} should succeed",
        )
        output = result.GetOutput()
        # Should contain both the base unwinder and the provider sections.
        self.assertIn("Base Unwinder", output)
        self.assertIn("PrefixPassThroughProvider", output)

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_bt_provider_range_not_starting_at_zero(self):
        """
        Test that 'bt --provider <id>-<id>' works when the range doesn't
        include the base unwinder (provider 0).
        """
        self.build()

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source)
        )

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

        # Range that only includes the provider, not the base unwinder.
        fid = self.FIRST_PROVIDER_FRAME_LIST_ID
        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            f"bt --provider {fid}-{fid}", result
        )
        self.assertTrue(
            result.Succeeded(),
            f"bt --provider {fid}-{fid} should succeed",
        )
        output = result.GetOutput()
        self.assertNotIn("Base Unwinder", output)
        self.assertIn("PrefixPassThroughProvider", output)
        self.assertIn("my_custom_baz", output)

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_bt_provider_range_with_to_separator(self):
        """
        Test that 'bt --provider 0 to <id>' works with the 'to' separator.
        """
        self.build()

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source)
        )

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

        # Use 'to' separator instead of '-'.
        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(
            f"bt --provider '0 to {self.FIRST_PROVIDER_FRAME_LIST_ID}'", result
        )
        self.assertTrue(
            result.Succeeded(),
            f"bt --provider '0 to {self.FIRST_PROVIDER_FRAME_LIST_ID}' should succeed",
        )
        output = result.GetOutput()
        self.assertIn("Base Unwinder", output)
        self.assertIn("PrefixPassThroughProvider", output)

    def test_bt_provider_invalid_id(self):
        """
        Test that 'bt --provider <invalid>' fails with an error when the
        provider ID doesn't exist.
        """
        self.build()

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source)
        )

        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("bt --provider 999", result)
        self.assertFalse(result.Succeeded(), "bt --provider 999 should fail")

    def test_bt_provider_invalid_range(self):
        """
        Test that 'bt --provider N-M' where N > M fails with an error
        about an invalid range.
        """
        self.build()

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source)
        )

        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("bt --provider 5-2", result)
        self.assertFalse(result.Succeeded(), "bt --provider 5-2 should fail")
        self.assertIn("invalid provider range", result.GetError())

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_bt_provider_star_shows_all(self):
        """
        Test that 'bt --provider *' shows all providers including the
        base unwinder and any registered providers.
        """
        self.build()

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source)
        )

        script_path = os.path.join(self.getSourceDir(), "frame_provider.py")
        self.runCmd("command script import " + script_path)

        error = lldb.SBError()
        target.RegisterScriptedFrameProvider(
            "frame_provider.PrefixPassThroughProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Should register provider: {error}")

        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("bt --provider '*'", result)
        self.assertTrue(result.Succeeded(), "bt --provider '*' should succeed")
        output = result.GetOutput()
        self.assertIn("Base Unwinder", output)
        self.assertIn("PrefixPassThroughProvider", output)

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_bt_provider_all_shows_all(self):
        """
        Test that 'bt --provider all' shows all providers including the
        base unwinder and any registered providers.
        """
        self.build()

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source)
        )

        script_path = os.path.join(self.getSourceDir(), "frame_provider.py")
        self.runCmd("command script import " + script_path)

        error = lldb.SBError()
        target.RegisterScriptedFrameProvider(
            "frame_provider.PrefixPassThroughProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Should register provider: {error}")

        target.RegisterScriptedFrameProvider(
            "frame_provider.UpperCasePassThroughProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Should register upper provider: {error}")

        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("bt --provider all", result)
        self.assertTrue(result.Succeeded(), "bt --provider all should succeed")
        output = result.GetOutput()
        self.assertIn("Base Unwinder", output)
        self.assertIn("PrefixPassThroughProvider", output)
        self.assertIn("UpperCasePassThroughProvider", output)

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_bt_provider_multiple_providers(self):
        """
        Test 'bt --provider' with two chained providers. Register
        PrefixPassThroughProvider (adds 'my_custom_' prefix) then
        UpperCasePassThroughProvider (upper-cases names). Verify each
        provider's view is correct and that ranges across providers work.
        """
        self.build()

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source)
        )

        script_path = os.path.join(self.getSourceDir(), "frame_provider.py")
        self.runCmd("command script import " + script_path)

        # Register first provider: adds 'my_custom_' prefix.
        error = lldb.SBError()
        target.RegisterScriptedFrameProvider(
            "frame_provider.PrefixPassThroughProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Should register prefix provider: {error}")

        # Register second provider: upper-cases everything.
        target.RegisterScriptedFrameProvider(
            "frame_provider.UpperCasePassThroughProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Should register upper provider: {error}")

        # bt --provider 0: base unwinder, original names.
        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("bt --provider 0", result)
        self.assertTrue(result.Succeeded())
        output = result.GetOutput()
        self.assertIn("Base Unwinder", output)
        self.assertIn("baz", output)

        # bt --provider all: should show all providers.
        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("bt --provider all", result)
        self.assertTrue(result.Succeeded())
        output = result.GetOutput()
        self.assertIn("Base Unwinder", output)
        self.assertIn("UpperCasePassThroughProvider", output)
        # UpperCase wraps Prefix (registration order is preserved), so the
        # outermost output should have fully upper-cased prefixed names.
        self.assertIn("MY_CUSTOM_BAZ", output)

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_bt_provider_star_from_within_provider(self):
        """
        Test that running 'bt --provider *' re-entrantly from within a
        scripted frame provider's get_frame_at_index does not deadlock
        or crash.

        BtProviderStarProvider runs 'bt --provider *' on its first
        get_frame_at_index call and stores the output, then passes through
        all frames with a 'reentrant_' prefix. We verify:
        1. The provider completes without hanging.
        2. Frame 0 has the 'reentrant_' prefix.
        """
        self.build()

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source)
        )

        script_path = os.path.join(self.getSourceDir(), "frame_provider.py")
        self.runCmd("command script import " + script_path)

        error = lldb.SBError()
        provider_id = target.RegisterScriptedFrameProvider(
            "frame_provider.BtProviderStarProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(
            error.Success(), f"Should register provider successfully: {error}"
        )
        self.assertNotEqual(provider_id, 0, "Provider ID should be non-zero")

        # Access frame 0 to trigger the provider (and the re-entrant bt call).
        # Only frame 0 is checked because the re-entrant HandleCommand resets
        # the selected frame on the unwinder list, which corrupts subsequent
        # frame lookups via inlined depth adjustments.
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(
            frame.GetFunctionName(),
            "reentrant_baz",
            "Frame 0 should be 'reentrant_baz' after provider",
        )

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

        # No frame should contain 'danger_will_robinson_' — that would mean
        # the provider was handed its own output list instead of the parent.
        num_frames = thread.GetNumFrames()
        for i in range(num_frames):
            frame = thread.GetFrameAtIndex(i)
            actual = frame.GetFunctionName()
            self.assertFalse(
                actual.startswith("danger_will_robinson_"),
                f"Frame {i}: provider got its own output list "
                f"(expected bare parent frames, got '{actual}')",
            )
