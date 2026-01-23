"""
Test scripted frame provider functionality.
"""

import os

import lldb
import lldbsuite.test.lldbplatformutil as lldbplatformutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil

class ScriptedFrameProviderTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.source = "main.cpp"

    def test_replace_all_frames(self):
        """Test that we can replace the entire stack."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source), only_one_thread=False
        )

        # Import the test frame provider.
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        # Attach the Replace provider.
        error = lldb.SBError()
        provider_id = target.RegisterScriptedFrameProvider(
            "test_frame_providers.ReplaceFrameProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Failed to register provider: {error}")
        self.assertNotEqual(provider_id, 0, "Provider ID should be non-zero")

        # Verify we have exactly 3 synthetic frames.
        self.assertEqual(thread.GetNumFrames(), 3, "Should have 3 synthetic frames")

        # Verify frame indices and PCs (dictionary-based frames don't have custom function names).
        frame0 = thread.GetFrameAtIndex(0)
        self.assertIsNotNone(frame0)
        self.assertEqual(frame0.GetPC(), 0x1000)

        frame1 = thread.GetFrameAtIndex(1)
        self.assertIsNotNone(frame1)
        self.assertIn("thread_func", frame1.GetFunctionName())

        frame2 = thread.GetFrameAtIndex(2)
        self.assertIsNotNone(frame2)
        self.assertEqual(frame2.GetPC(), 0x3000)

    def test_prepend_frames(self):
        """Test that we can add frames before real stack."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source), only_one_thread=False
        )

        # Get original frame count and PC.
        original_frame_count = thread.GetNumFrames()
        self.assertGreaterEqual(
            original_frame_count, 2, "Should have at least 2 real frames"
        )

        # Import and attach Prepend provider.
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        error = lldb.SBError()
        provider_id = target.RegisterScriptedFrameProvider(
            "test_frame_providers.PrependFrameProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Failed to register provider: {error}")
        self.assertNotEqual(provider_id, 0, "Provider ID should be non-zero")

        # Verify we have 2 more frames.
        new_frame_count = thread.GetNumFrames()
        self.assertEqual(new_frame_count, original_frame_count + 2)

        # Verify first 2 frames are synthetic (check PCs, not function names).
        frame0 = thread.GetFrameAtIndex(0)
        self.assertEqual(frame0.GetPC(), 0x9000)

        frame1 = thread.GetFrameAtIndex(1)
        self.assertEqual(frame1.GetPC(), 0xA000)

        # Verify frame 2 is the original real frame 0.
        frame2 = thread.GetFrameAtIndex(2)
        self.assertIn("thread_func", frame2.GetFunctionName())

    def test_append_frames(self):
        """Test that we can add frames after real stack."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source), only_one_thread=False
        )

        # Get original frame count.
        original_frame_count = thread.GetNumFrames()

        # Import and attach Append provider.
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        error = lldb.SBError()
        provider_id = target.RegisterScriptedFrameProvider(
            "test_frame_providers.AppendFrameProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Failed to register provider: {error}")
        self.assertNotEqual(provider_id, 0, "Provider ID should be non-zero")

        # Verify we have 1 more frame.
        new_frame_count = thread.GetNumFrames()
        self.assertEqual(new_frame_count, original_frame_count + 1)

        # Verify first frames are still real.
        frame0 = thread.GetFrameAtIndex(0)
        self.assertIn("thread_func", frame0.GetFunctionName())

        frame_n_plus_1 = thread.GetFrameAtIndex(new_frame_count - 1)
        self.assertEqual(frame_n_plus_1.GetPC(), 0x10)

    def test_scripted_frame_objects(self):
        """Test that provider can return ScriptedFrame objects."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source), only_one_thread=False
        )

        # Import the provider that returns ScriptedFrame objects.
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        error = lldb.SBError()
        provider_id = target.RegisterScriptedFrameProvider(
            "test_frame_providers.ScriptedFrameObjectProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Failed to register provider: {error}")
        self.assertNotEqual(provider_id, 0, "Provider ID should be non-zero")

        # Verify we have 5 frames.
        self.assertEqual(
            thread.GetNumFrames(), 5, "Should have 5 custom scripted frames"
        )

        # Verify frame properties from CustomScriptedFrame.
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

    def test_applies_to_thread(self):
        """Test that applies_to_thread filters which threads get the provider."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source), only_one_thread=False
        )

        # We should have at least 2 threads (worker threads) at the breakpoint.
        num_threads = process.GetNumThreads()
        self.assertGreaterEqual(
            num_threads, 2, "Should have at least 2 threads at breakpoint"
        )

        # Import the test frame provider.
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        # Collect original thread info before applying provider.
        thread_info = {}
        for i in range(num_threads):
            t = process.GetThreadAtIndex(i)
            thread_info[t.GetIndexID()] = {
                "frame_count": t.GetNumFrames(),
                "pc": t.GetFrameAtIndex(0).GetPC(),
            }

        # Register the ThreadFilterFrameProvider which only applies to thread ID 1.
        error = lldb.SBError()
        provider_id = target.RegisterScriptedFrameProvider(
            "test_frame_providers.ThreadFilterFrameProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Failed to register provider: {error}")
        self.assertNotEqual(provider_id, 0, "Provider ID should be non-zero")

        # Check each thread.
        thread_id_1_found = False
        # On ARM32, FixCodeAddress clears bit 0, so synthetic PCs get modified.
        is_arm_32bit = lldbplatformutil.getArchitecture() == "arm"
        expected_synthetic_pc = 0xFFFE if is_arm_32bit else 0xFFFF

        for i in range(num_threads):
            t = process.GetThreadAtIndex(i)
            thread_id = t.GetIndexID()

            if thread_id == 1:
                # Thread with ID 1 should have synthetic frame.
                thread_id_1_found = True
                self.assertEqual(
                    t.GetNumFrames(),
                    1,
                    f"Thread with ID 1 should have 1 synthetic frame",
                )
                self.assertEqual(
                    t.GetFrameAtIndex(0).GetPC(),
                    expected_synthetic_pc,
                    f"Thread with ID 1 should have synthetic PC {expected_synthetic_pc:#x}",
                )
            else:
                # Other threads should keep their original frames.
                self.assertEqual(
                    t.GetNumFrames(),
                    thread_info[thread_id]["frame_count"],
                    f"Thread with ID {thread_id} should not be affected by provider",
                )
                self.assertEqual(
                    t.GetFrameAtIndex(0).GetPC(),
                    thread_info[thread_id]["pc"],
                    f"Thread with ID {thread_id} should have its original PC",
                )

        # We should have found at least one thread with ID 1.
        self.assertTrue(
            thread_id_1_found,
            "Should have found a thread with ID 1 to test filtering",
        )

    def test_remove_frame_provider_by_id(self):
        """Test that RemoveScriptedFrameProvider removes a specific provider by ID."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source), only_one_thread=False
        )

        # Import the test frame providers.
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        # Get original frame count.
        original_frame_count = thread.GetNumFrames()
        original_pc = thread.GetFrameAtIndex(0).GetPC()

        # Register the first provider and get its ID.
        error = lldb.SBError()
        provider_id_1 = target.RegisterScriptedFrameProvider(
            "test_frame_providers.ReplaceFrameProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Failed to register provider 1: {error}")

        # Verify first provider is active (3 synthetic frames).
        self.assertEqual(thread.GetNumFrames(), 3, "Should have 3 synthetic frames")
        self.assertEqual(
            thread.GetFrameAtIndex(0).GetPC(), 0x1000, "Should have first provider's PC"
        )

        # Register a second provider and get its ID.
        provider_id_2 = target.RegisterScriptedFrameProvider(
            "test_frame_providers.PrependFrameProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Failed to register provider 2: {error}")

        # Verify IDs are different
        self.assertNotEqual(
            provider_id_1, provider_id_2, "Provider IDs should be unique"
        )

        # Now remove the first provider by ID
        result = target.RemoveScriptedFrameProvider(provider_id_1)
        self.assertSuccess(
            result, f"Should successfully remove provider with ID {provider_id_1}"
        )

        # After removing the first provider, the second provider should still be
        # active. The PrependFrameProvider adds 2 frames before the real stack.
        # Since ReplaceFrameProvider had 3 frames, and we removed it, we should now
        # have the original frames (from real stack) with PrependFrameProvider applied.
        new_frame_count = thread.GetNumFrames()
        self.assertEqual(
            new_frame_count,
            original_frame_count + 2,
            "Should have original frames + 2 prepended frames",
        )

        # First two frames should be from PrependFrameProvider.
        self.assertEqual(
            thread.GetFrameAtIndex(0).GetPC(),
            0x9000,
            "First frame should be from PrependFrameProvider",
        )
        self.assertEqual(
            thread.GetFrameAtIndex(1).GetPC(),
            0xA000,
            "Second frame should be from PrependFrameProvider",
        )

        # Remove the second provider.
        result = target.RemoveScriptedFrameProvider(provider_id_2)
        self.assertSuccess(
            result, f"Should successfully remove provider with ID {provider_id_2}"
        )

        # After removing both providers, frames should be back to original.
        self.assertEqual(
            thread.GetNumFrames(),
            original_frame_count,
            "Should restore original frame count",
        )
        self.assertEqual(
            thread.GetFrameAtIndex(0).GetPC(),
            original_pc,
            "Should restore original PC",
        )

        # Try to remove a provider that doesn't exist.
        result = target.RemoveScriptedFrameProvider(999999)
        self.assertTrue(result.Fail(), "Should fail to remove non-existent provider")

    def test_circular_dependency_fix(self):
        """Test that accessing input_frames in __init__ doesn't cause circular dependency.

        This test verifies the fix for the circular dependency issue where:
        1. Thread::GetStackFrameList() creates the frame provider
        2. Provider's __init__ accesses input_frames and calls methods on frames
        3. SBFrame methods trigger ExecutionContextRef::GetFrameSP()
        4. Before the fix: GetFrameSP() would call Thread::GetStackFrameList() again -> circular dependency!
        5. After the fix: GetFrameSP() uses the remembered frame list -> no circular dependency

        The fix works by:
        - StackFrame stores m_frame_list_wp (weak pointer to originating list)
        - ExecutionContextRef stores m_frame_list_wp when created from a frame
        - ExecutionContextRef::GetFrameSP() tries the remembered list first before asking the thread
        """
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source), only_one_thread=False
        )

        # Get original frame count and PC.
        original_frame_count = thread.GetNumFrames()
        original_pc = thread.GetFrameAtIndex(0).GetPC()
        self.assertGreaterEqual(
            original_frame_count, 2, "Should have at least 2 real frames"
        )

        # Import the provider that accesses input frames in __init__.
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        # Register the CircularDependencyTestProvider.
        # Before the fix, this would crash or hang due to circular dependency.
        error = lldb.SBError()
        provider_id = target.RegisterScriptedFrameProvider(
            "test_frame_providers.CircularDependencyTestProvider",
            lldb.SBStructuredData(),
            error,
        )

        # If we get here without crashing, the fix is working!
        self.assertTrue(error.Success(), f"Failed to register provider: {error}")
        self.assertNotEqual(provider_id, 0, "Provider ID should be non-zero")

        # Verify the provider worked correctly,
        # Should have 1 synthetic frame + all original frames.
        new_frame_count = thread.GetNumFrames()
        self.assertEqual(
            new_frame_count,
            original_frame_count + 1,
            "Should have original frames + 1 synthetic frame",
        )

        # On ARM32, FixCodeAddress clears bit 0, so synthetic PCs get modified.
        is_arm_32bit = lldbplatformutil.getArchitecture() == "arm"
        expected_synthetic_pc = 0xDEADBEEE if is_arm_32bit else 0xDEADBEEF

        # First frame should be synthetic.
        frame0 = thread.GetFrameAtIndex(0)
        self.assertIsNotNone(frame0)
        self.assertEqual(
            frame0.GetPC(),
            expected_synthetic_pc,
            f"First frame should be synthetic frame with PC {expected_synthetic_pc:#x}",
        )

        # Second frame should be the original first frame.
        frame1 = thread.GetFrameAtIndex(1)
        self.assertIsNotNone(frame1)
        self.assertEqual(
            frame1.GetPC(),
            original_pc,
            "Second frame should be original first frame",
        )

        # Verify we can still call methods on frames (no circular dependency!).
        for i in range(min(3, new_frame_count)):
            frame = thread.GetFrameAtIndex(i)
            self.assertIsNotNone(frame)
            # These calls should not trigger circular dependency.
            pc = frame.GetPC()
            self.assertNotEqual(pc, 0, f"Frame {i} should have valid PC")

    def test_python_source_frames(self):
        """Test that frames can point to Python source files and display properly."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source), only_one_thread=False
        )

        # Get original frame count.
        original_frame_count = thread.GetNumFrames()
        self.assertGreaterEqual(
            original_frame_count, 2, "Should have at least 2 real frames"
        )

        # Import the provider.
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        # Register the PythonSourceFrameProvider.
        error = lldb.SBError()
        provider_id = target.RegisterScriptedFrameProvider(
            "test_frame_providers.PythonSourceFrameProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Failed to register provider: {error}")
        self.assertNotEqual(provider_id, 0, "Provider ID should be non-zero")

        # Verify we have 3 more frames (Python frames).
        new_frame_count = thread.GetNumFrames()
        self.assertEqual(
            new_frame_count,
            original_frame_count + 3,
            "Should have original frames + 3 Python frames",
        )

        # Verify first three frames are Python source frames.
        frame0 = thread.GetFrameAtIndex(0)
        self.assertIsNotNone(frame0)
        self.assertEqual(
            frame0.GetFunctionName(),
            "compute_fibonacci",
            "First frame should be compute_fibonacci",
        )
        self.assertTrue(frame0.IsSynthetic(), "Frame should be marked as synthetic")
        # PC-less frames should show invalid address and not crash.
        self.assertEqual(
            frame0.GetPC(),
            lldb.LLDB_INVALID_ADDRESS,
            "PC-less frame should have LLDB_INVALID_ADDRESS",
        )

        self.assertEqual(
            frame0.GetFP(),
            lldb.LLDB_INVALID_ADDRESS,
            "PC-less frame FP should return LLDB_INVALID_ADDRESS",
        )
        self.assertEqual(
            frame0.GetSP(),
            lldb.LLDB_INVALID_ADDRESS,
            "PC-less frame SP should return LLDB_INVALID_ADDRESS",
        )
        self.assertEqual(
            frame0.GetCFA(),
            0,
            "PC-less frame CFA should return 0",
        )

        frame1 = thread.GetFrameAtIndex(1)
        self.assertIsNotNone(frame1)
        self.assertEqual(
            frame1.GetFunctionName(),
            "process_data",
            "Second frame should be process_data",
        )
        self.assertTrue(frame1.IsSynthetic(), "Frame should be marked as synthetic")

        frame2 = thread.GetFrameAtIndex(2)
        self.assertIsNotNone(frame2)
        self.assertEqual(frame2.GetFunctionName(), "main", "Third frame should be main")
        self.assertTrue(frame2.IsSynthetic(), "Frame should be marked as synthetic")

        # Verify line entry information is present.
        line_entry0 = frame0.GetLineEntry()
        self.assertTrue(line_entry0.IsValid(), "Frame 0 should have a valid line entry")
        self.assertEqual(line_entry0.GetLine(), 7, "Frame 0 should point to line 7")
        file_spec0 = line_entry0.GetFileSpec()
        self.assertTrue(file_spec0.IsValid(), "Frame 0 should have valid file spec")
        self.assertEqual(
            file_spec0.GetFilename(),
            "python_helper.py",
            "Frame 0 should point to python_helper.py",
        )

        line_entry1 = frame1.GetLineEntry()
        self.assertTrue(line_entry1.IsValid(), "Frame 1 should have a valid line entry")
        self.assertEqual(line_entry1.GetLine(), 16, "Frame 1 should point to line 16")

        line_entry2 = frame2.GetLineEntry()
        self.assertTrue(line_entry2.IsValid(), "Frame 2 should have a valid line entry")
        self.assertEqual(line_entry2.GetLine(), 27, "Frame 2 should point to line 27")

        # Verify the frames display properly in backtrace.
        # This tests that PC-less frames don't show 0xffffffffffffffff.
        self.runCmd("bt")
        output = self.res.GetOutput()

        # Should show function names.
        self.assertIn("compute_fibonacci", output)
        self.assertIn("process_data", output)
        self.assertIn("main", output)

        # Should show Python file.
        self.assertIn("python_helper.py", output)

        # Should show line numbers.
        self.assertIn(":7", output)  # compute_fibonacci line.
        self.assertIn(":16", output)  # process_data line.
        self.assertIn(":27", output)  # main line.

        # Should NOT show invalid address (0xffffffffffffffff).
        self.assertNotIn("0xffffffffffffffff", output.lower())

        # Verify frame 3 is the original real frame 0.
        frame3 = thread.GetFrameAtIndex(3)
        self.assertIsNotNone(frame3)
        self.assertIn("thread_func", frame3.GetFunctionName())

    def test_valid_pc_no_module_frames(self):
        """Test that frames with valid PC but no module display correctly in backtrace."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source), only_one_thread=False
        )

        # Get original frame count.
        original_frame_count = thread.GetNumFrames()
        self.assertGreaterEqual(
            original_frame_count, 2, "Should have at least 2 real frames"
        )

        # Import the provider.
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        # Register the ValidPCNoModuleFrameProvider.
        error = lldb.SBError()
        provider_id = target.RegisterScriptedFrameProvider(
            "test_frame_providers.ValidPCNoModuleFrameProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Failed to register provider: {error}")
        self.assertNotEqual(provider_id, 0, "Provider ID should be non-zero")

        # Verify we have 2 more frames (the synthetic frames).
        new_frame_count = thread.GetNumFrames()
        self.assertEqual(
            new_frame_count,
            original_frame_count + 2,
            "Should have original frames + 2 synthetic frames",
        )

        # Verify first two frames have valid PCs and function names.
        frame0 = thread.GetFrameAtIndex(0)
        self.assertIsNotNone(frame0)
        self.assertEqual(
            frame0.GetFunctionName(),
            "unknown_function_1",
            "First frame should be unknown_function_1",
        )
        self.assertTrue(frame0.IsSynthetic(), "Frame should be marked as synthetic")
        self.assertEqual(
            frame0.GetPC(), 0x1234000, "First frame should have PC 0x1234000"
        )

        frame1 = thread.GetFrameAtIndex(1)
        self.assertIsNotNone(frame1)
        self.assertEqual(
            frame1.GetFunctionName(),
            "unknown_function_2",
            "Second frame should be unknown_function_2",
        )
        self.assertTrue(frame1.IsSynthetic(), "Frame should be marked as synthetic")
        self.assertEqual(
            frame1.GetPC(), 0x5678000, "Second frame should have PC 0x5678000"
        )

        # Verify the frames display properly in backtrace.
        # The backtrace should show the PC values without crashing or displaying
        # invalid addresses like 0xffffffffffffffff.
        self.runCmd("bt")
        output = self.res.GetOutput()

        # Should show function names.
        self.assertIn("unknown_function_1", output)
        self.assertIn("unknown_function_2", output)

        # Should show PC addresses in hex format.
        self.assertIn("1234000", output)
        self.assertIn("5678000", output)

        # Verify PC and function name are properly separated by space.
        self.assertIn("1234000 unknown_function_1", output)
        self.assertIn("5678000 unknown_function_2", output)

        # Should NOT show invalid address.
        self.assertNotIn("ffffff", output.lower())

        # Verify frame 2 is the original real frame 0.
        frame2 = thread.GetFrameAtIndex(2)
        self.assertIsNotNone(frame2)
        self.assertIn("thread_func", frame2.GetFunctionName())

    def test_chained_frame_providers(self):
        """Test that multiple frame providers chain together."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec(self.source), only_one_thread=False
        )

        # Get original frame count.
        original_frame_count = thread.GetNumFrames()
        self.assertGreaterEqual(
            original_frame_count, 2, "Should have at least 2 real frames"
        )

        # Import the test frame providers.
        script_path = os.path.join(self.getSourceDir(), "test_frame_providers.py")
        self.runCmd("command script import " + script_path)

        # Register 3 providers with different priorities.
        # Each provider adds 1 frame at the beginning.
        error = lldb.SBError()

        # Provider 1: Priority 10 - adds "foo" frame
        provider_id_1 = target.RegisterScriptedFrameProvider(
            "test_frame_providers.AddFooFrameProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Failed to register foo provider: {error}")

        # Provider 2: Priority 20 - adds "bar" frame
        provider_id_2 = target.RegisterScriptedFrameProvider(
            "test_frame_providers.AddBarFrameProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Failed to register bar provider: {error}")

        # Provider 3: Priority 30 - adds "baz" frame
        provider_id_3 = target.RegisterScriptedFrameProvider(
            "test_frame_providers.AddBazFrameProvider",
            lldb.SBStructuredData(),
            error,
        )
        self.assertTrue(error.Success(), f"Failed to register baz provider: {error}")

        # Verify we have 3 more frames (one from each provider).
        new_frame_count = thread.GetNumFrames()
        self.assertEqual(
            new_frame_count,
            original_frame_count + 3,
            "Should have original frames + 3 chained frames",
        )

        # Verify the chaining order: baz, bar, foo, then real frames.
        # Since priority is lower = higher, the order should be:
        # Provider 1 (priority 10) transforms real frames first -> adds "foo"
        # Provider 2 (priority 20) transforms Provider 1's output -> adds "bar"
        # Provider 3 (priority 30) transforms Provider 2's output -> adds "baz"
        # So final stack is: baz, bar, foo, real frames...

        frame0 = thread.GetFrameAtIndex(0)
        self.assertIsNotNone(frame0)
        self.assertEqual(
            frame0.GetFunctionName(),
            "baz",
            "Frame 0 should be 'baz' from last provider in chain",
        )
        self.assertEqual(frame0.GetPC(), 0xBAD)

        frame1 = thread.GetFrameAtIndex(1)
        self.assertIsNotNone(frame1)
        self.assertEqual(
            frame1.GetFunctionName(),
            "bar",
            "Frame 1 should be 'bar' from second provider in chain",
        )
        self.assertEqual(frame1.GetPC(), 0xBAB)

        frame2 = thread.GetFrameAtIndex(2)
        self.assertIsNotNone(frame2)
        self.assertEqual(
            frame2.GetFunctionName(),
            "foo",
            "Frame 2 should be 'foo' from first provider in chain",
        )
        self.assertEqual(frame2.GetPC(), 0xF00)

        # Frame 3 should be the original real frame 0.
        frame3 = thread.GetFrameAtIndex(3)
        self.assertIsNotNone(frame3)
        self.assertIn("thread_func", frame3.GetFunctionName())
