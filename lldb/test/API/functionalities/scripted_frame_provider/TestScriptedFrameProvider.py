"""
Test scripted frame provider functionality with all merge strategies.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ScriptedFrameProviderTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.source = "main.c"

    def test_replace_strategy(self):
        """Test that Replace strategy replaces entire stack."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Import the test frame provider
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        # Attach the Replace provider
        target.RegisterScriptedFrameProvider(
            "test_frame_providers.ReplaceFrameProvider", lldb.SBStructuredData()
        )

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source)
        )

        # Verify we have exactly 3 synthetic frames
        self.assertEqual(thread.GetNumFrames(), 3, "Should have 3 synthetic frames")

        # Verify frame indices and PCs (dictionary-based frames don't have custom function names)
        frame0 = thread.GetFrameAtIndex(0)
        self.assertIsNotNone(frame0)
        self.assertEqual(frame0.GetPC(), 0x1000)

        frame1 = thread.GetFrameAtIndex(1)
        self.assertIsNotNone(frame1)
        self.assertIn("main", frame1.GetFunctionName())

        frame2 = thread.GetFrameAtIndex(2)
        self.assertIsNotNone(frame2)
        self.assertEqual(frame2.GetPC(), 0x3000)

    def test_prepend_strategy(self):
        """Test that Prepend strategy adds frames before real stack."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source)
        )

        # Get original frame count and PC
        original_frame_count = thread.GetNumFrames()
        self.assertGreaterEqual(
            original_frame_count, 2, "Should have at least 2 real frames"
        )
        original_frame_0_pc = thread.GetFrameAtIndex(0).GetPC()

        # Import and attach Prepend provider
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        target.RegisterScriptedFrameProvider(
            "test_frame_providers.PrependFrameProvider", lldb.SBStructuredData()
        )

        # Verify we have 2 more frames
        new_frame_count = thread.GetNumFrames()
        self.assertEqual(new_frame_count, original_frame_count + 2)

        # Verify first 2 frames are synthetic (check PCs, not function names)
        frame0 = thread.GetFrameAtIndex(0)
        self.assertEqual(frame0.GetPC(), 0x9000)

        frame1 = thread.GetFrameAtIndex(1)
        self.assertEqual(frame1.GetPC(), 0xA000)

        # Verify frame 2 is the original real frame 0
        frame2 = thread.GetFrameAtIndex(2)
        self.assertIn("foo", frame2.GetFunctionName())

    def test_append_strategy(self):
        """Test that Append strategy adds frames after real stack."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source)
        )

        # Get original frame count
        original_frame_count = thread.GetNumFrames()

        # Import and attach Append provider
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        target.RegisterScriptedFrameProvider(
            "test_frame_providers.AppendFrameProvider", lldb.SBStructuredData()
        )

        # Verify we have 2 more frames
        new_frame_count = thread.GetNumFrames()
        self.assertEqual(new_frame_count, original_frame_count + 1)

        # Verify first frames are still real
        frame0 = thread.GetFrameAtIndex(0)
        self.assertIn("foo", frame0.GetFunctionName())

        frame_n_plus_1 = thread.GetFrameAtIndex(new_frame_count - 1)
        self.assertEqual(frame_n_plus_1.GetPC(), 0x10)

    def test_replace_by_index_strategy(self):
        """Test that ReplaceByIndex strategy replaces specific frames."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source)
        )

        # Get original frame count and info
        original_frame_count = thread.GetNumFrames()
        self.assertGreaterEqual(original_frame_count, 3, "Need at least 3 frames")

        original_frame_0_pc = thread.GetFrameAtIndex(0).GetPC()
        original_frame_1 = thread.GetFrameAtIndex(1)
        original_frame_1_name = original_frame_1.GetFunctionName()
        original_frame_2_pc = thread.GetFrameAtIndex(2).GetPC()

        # Import and attach ReplaceByIndex provider
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        target.RegisterScriptedFrameProvider(
            "test_frame_providers.ReplaceByIndexFrameProvider", lldb.SBStructuredData()
        )

        # Verify frame count unchanged
        self.assertEqual(
            thread.GetNumFrames(),
            original_frame_count,
            "Frame count should remain the same",
        )

        # Verify frame 0 is replaced (PC should match original since provider uses it)
        frame0 = thread.GetFrameAtIndex(0)
        self.assertIsNotNone(frame0)
        self.assertEqual(
            frame0.GetPC(), original_frame_0_pc, "Frame 0 should be replaced"
        )

        # Verify frame 1 is still the original (not replaced)
        frame1 = thread.GetFrameAtIndex(1)
        self.assertIsNotNone(frame1)
        self.assertEqual(
            frame1.GetFunctionName(),
            original_frame_1_name,
            "Frame 1 should remain unchanged",
        )

        # Verify frame 2 is replaced (PC should match original since provider uses it)
        frame2 = thread.GetFrameAtIndex(2)
        self.assertIsNotNone(frame2)
        self.assertEqual(
            frame2.GetPC(), original_frame_2_pc, "Frame 2 should be replaced"
        )

    def test_clear_frame_provider(self):
        """Test that clearing provider restores normal unwinding."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source)
        )

        # Get original state
        original_frame_0 = thread.GetFrameAtIndex(0)
        original_frame_0_name = original_frame_0.GetFunctionName()
        original_frame_count = thread.GetNumFrames()

        # Import and attach provider
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        target.RegisterScriptedFrameProvider(
            "test_frame_providers.ReplaceFrameProvider", lldb.SBStructuredData()
        )

        # Verify frames are synthetic (3 frames with specific PCs)
        self.assertEqual(thread.GetNumFrames(), 3)
        frame0 = thread.GetFrameAtIndex(0)
        self.assertEqual(frame0.GetPC(), 0x1000)

        # Clear the provider
        target.ClearScriptedFrameProvider()

        # Verify frames are back to normal
        self.assertEqual(thread.GetNumFrames(), original_frame_count)
        frame0 = thread.GetFrameAtIndex(0)
        self.assertEqual(
            frame0.GetFunctionName(),
            original_frame_0_name,
            "Should restore original frames after clearing provider",
        )

    def test_scripted_frame_objects(self):
        """Test that provider can return ScriptedFrame objects."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source)
        )

        # Import the provider that returns ScriptedFrame objects
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        target.RegisterScriptedFrameProvider(
            "test_frame_providers.ScriptedFrameObjectProvider", lldb.SBStructuredData()
        )

        # Verify we have 5 frames
        self.assertEqual(
            thread.GetNumFrames(), 5, "Should have 5 custom scripted frames"
        )

        # Verify frame properties from CustomScriptedFrame
        frame0 = thread.GetFrameAtIndex(0)
        self.assertIsNotNone(frame0)
        self.assertEqual(frame0.GetFunctionName(), "custom_scripted_frame_0")
        self.assertEqual(frame0.GetPC(), 0x5000)
        self.assertTrue(frame0.IsSynthetic(), "Frame should be marked as synthetic")

        frame1 = thread.GetFrameAtIndex(1)
        self.assertIsNotNone(frame1)
        self.assertEqual(frame1.GetPC(), 0x6000)

        frame2 = thread.GetFrameAtIndex(2)
        self.assertIsNotNone(frame2)
        self.assertEqual(frame2.GetFunctionName(), "custom_scripted_frame_2")
        self.assertEqual(frame2.GetPC(), 0x7000)
        self.assertTrue(frame2.IsSynthetic(), "Frame should be marked as synthetic")
