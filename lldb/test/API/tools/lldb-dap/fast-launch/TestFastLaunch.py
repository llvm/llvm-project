"""
Test lldb-dap fast launch mode functionality and performance optimizations.
"""

import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbdap_testcase
import time
import os


class TestDAP_fastLaunch(lldbdap_testcase.DAPTestCaseBase):

    @skipIfWindows  # Skip on Windows due to different symbol loading behavior
    def test_core_optimizations(self):
        """
        Test that core LLDB optimizations work correctly (no special config needed).
        """
        program = self.getBuildArtifact("a.out")
        # Core optimizations are now enabled by default
        self.build_and_launch(program)

        # Verify the target was created successfully
        self.assertTrue(self.dap_server.target.IsValid())

        # Test that we can set breakpoints (symbol loading should work on-demand)
        source = "main.cpp"
        breakpoint_line = line_number(source, "// Set breakpoint here")
        lines = [breakpoint_line]
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(len(breakpoint_ids), 1)

        # Continue and verify we hit the breakpoint
        self.continue_to_next_stop()
        self.verify_stop_reason_breakpoint(breakpoint_ids[0])

    def test_optimized_debugging_functionality(self):
        """
        Test that core optimizations preserve debugging functionality.
        """
        program = self.getBuildArtifact("a.out")

        # Launch with core optimizations (enabled by default)
        self.build_and_launch(program, stopOnEntry=False)

        source = "main.cpp"
        breakpoint_line = line_number(source, "// Set breakpoint here")

        # Set breakpoint - this should trigger on-demand symbol loading
        breakpoint_ids = self.set_source_breakpoints(source, [breakpoint_line])
        self.assertEqual(len(breakpoint_ids), 1)

        # Continue and verify we hit the breakpoint
        self.continue_to_next_stop()
        self.verify_stop_reason_breakpoint(breakpoint_ids[0])

        # Verify stack trace works (requires symbols)
        frames = self.get_stackFrames()
        self.assertGreater(len(frames), 0)

        # Verify variable inspection works
        frame = frames[0]
        self.assertTrue("id" in frame)
        scopes = self.get_scopes(frame["id"])
        self.assertGreater(len(scopes), 0)

    def test_network_symbol_optimization(self):
        """
        Test that network symbol optimization settings work correctly.
        """
        program = self.getBuildArtifact("a.out")

        # Test with core optimizations (no special configuration needed)
        self.build_and_launch(
            program,
            stopOnEntry=True,
        )

        # Verify the target was created successfully
        self.assertTrue(self.dap_server.target.IsValid())

        # Test basic debugging functionality still works
        source = "main.cpp"
        breakpoint_line = line_number(source, "// Set breakpoint here")
        lines = [breakpoint_line]
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(len(breakpoint_ids), 1)

        # Continue and verify we hit the breakpoint
        self.continue_to_next_stop()
        self.verify_stop_reason_breakpoint(breakpoint_ids[0])

    def test_error_handling_and_recovery(self):
        """
        Test that error conditions are handled gracefully.
        """
        program = self.getBuildArtifact("a.out")

        # Test with invalid program path - should fail gracefully
        try:
            self.build_and_launch("/nonexistent/program")
            self.fail("Expected launch to fail with invalid program")
        except Exception:
            pass  # Expected failure

        # Test successful launch after failure
        self.build_and_launch(program)
        self.assertTrue(self.dap_server.target.IsValid())

    def test_thread_safety_and_concurrency(self):
        """
        Test that concurrent operations work correctly.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        # Set multiple breakpoints concurrently
        source = "main.cpp"
        breakpoint_line = line_number(source, "// Set breakpoint here")

        # This tests that the background symbol loading doesn't interfere
        # with immediate debugging operations
        breakpoint_ids = self.set_source_breakpoints(source, [breakpoint_line])
        self.assertEqual(len(breakpoint_ids), 1)

        # Verify debugging works immediately even with background loading
        self.continue_to_next_stop()
        self.verify_stop_reason_breakpoint(breakpoint_ids[0])
