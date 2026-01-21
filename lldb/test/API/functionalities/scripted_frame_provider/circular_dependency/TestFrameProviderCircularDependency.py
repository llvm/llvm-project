"""
Test that frame providers wouldn't cause a hang due to a circular dependency
during its initialization.
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil

class FrameProviderCircularDependencyTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.source = "main.c"

    @expectedFailureAll(oslist=["linux"], archs=["arm$"])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test_circular_dependency_with_function_replacement(self):
        """
        Test the circular dependency fix with a provider that replaces function names.
        """
        self.build()

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, "Target should be valid")

        bkpt = target.BreakpointCreateBySourceRegex(
            "break here", lldb.SBFileSpec(self.source)
        )
        self.assertTrue(bkpt.IsValid(), "Breakpoint should be valid")
        self.assertEqual(bkpt.GetNumLocations(), 1, "Should have 1 breakpoint location")

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, "Process should be valid")
        self.assertEqual(
            process.GetState(), lldb.eStateStopped, "Process should be stopped"
        )

        thread = process.GetSelectedThread()
        self.assertTrue(thread.IsValid(), "Thread should be valid")

        frame0 = thread.GetFrameAtIndex(0)
        self.assertIn("bar", frame0.GetFunctionName(), "Should be stopped in bar()")

        original_frame_count = thread.GetNumFrames()
        self.assertGreaterEqual(
            original_frame_count, 3, "Should have at least 3 frames: bar, foo, main"
        )

        frame_names = [thread.GetFrameAtIndex(i).GetFunctionName() for i in range(3)]
        self.assertEqual(frame_names[0], "bar", "Frame 0 should be bar")
        self.assertEqual(frame_names[1], "foo", "Frame 1 should be foo")
        self.assertEqual(frame_names[2], "main", "Frame 2 should be main")

        script_path = os.path.join(self.getSourceDir(), "frame_provider.py")
        self.runCmd("command script import " + script_path)

        # Register the frame provider that accesses input_frames.
        # Before the fix, this registration would trigger the circular dependency:
        # - Thread::GetStackFrameList() creates provider
        # - Provider's get_frame_at_index() accesses input_frames[0]
        # - Calls frame.GetFunctionName() -> ExecutionContextRef::GetFrameSP()
        # - Before fix: Calls Thread::GetStackFrameList() again -> CIRCULAR!
        # - After fix: Uses remembered m_frame_list_wp -> Works!
        error = lldb.SBError()
        provider_id = target.RegisterScriptedFrameProvider(
            "frame_provider.ScriptedFrameObjectProvider",
            lldb.SBStructuredData(),
            error,
        )

        # If we reach here without crashing/hanging, the fix is working!
        self.assertTrue(
            error.Success(),
            f"Should successfully register provider (if this fails, circular dependency!): {error}",
        )
        self.assertNotEqual(provider_id, 0, "Provider ID should be non-zero")

        # Verify the provider is working correctly.
        # Frame count should be unchanged (we're replacing frames, not adding).
        new_frame_count = thread.GetNumFrames()
        self.assertEqual(
            new_frame_count,
            original_frame_count,
            "Frame count should be unchanged (replacement, not addition)",
        )

        # Verify that "bar" was replaced with "baz".
        frame0_new = thread.GetFrameAtIndex(0)
        self.assertIsNotNone(frame0_new, "Frame 0 should exist")
        self.assertEqual(
            frame0_new.GetFunctionName(),
            "baz",
            "Frame 0 function should be replaced: bar -> baz",
        )

        # Verify other frames are unchanged.
        frame1_new = thread.GetFrameAtIndex(1)
        self.assertEqual(
            frame1_new.GetFunctionName(), "foo", "Frame 1 should still be foo"
        )

        frame2_new = thread.GetFrameAtIndex(2)
        self.assertEqual(
            frame2_new.GetFunctionName(), "main", "Frame 2 should still be main"
        )

        # Verify we can call methods on all frames (no circular dependency!).
        for i in range(new_frame_count):
            frame = thread.GetFrameAtIndex(i)
            self.assertIsNotNone(frame, f"Frame {i} should exist")
            # These calls should not trigger circular dependency.
            pc = frame.GetPC()
            self.assertNotEqual(pc, 0, f"Frame {i} should have valid PC")
            func_name = frame.GetFunctionName()
            self.assertIsNotNone(func_name, f"Frame {i} should have function name")
