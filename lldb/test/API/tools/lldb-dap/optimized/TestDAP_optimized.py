"""
Test lldb-dap variables/stackTrace request for optimized code
"""

import dap_server
import lldbdap_testcase
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestDAP_optimized(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    def test_stack_frame_name(self):
        """Test optimized frame has special name suffix."""
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        breakpoint_line = line_number(source, "// breakpoint 1")
        lines = [breakpoint_line]
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )
        self.continue_to_breakpoints(breakpoint_ids)
        leaf_frame = self.dap_server.get_stackFrame(frameIndex=0)
        self.assertTrue(leaf_frame["name"].endswith(" [opt]"))
        parent_frame = self.dap_server.get_stackFrame(frameIndex=1)
        self.assertTrue(parent_frame["name"].endswith(" [opt]"))

    @skipIfAsan # On ASAN builds this test intermittently fails https://github.com/llvm/llvm-project/issues/111061
    @skipIfWindows
    def test_optimized_variable(self):
        """Test optimized variable value contains error."""
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        breakpoint_line = line_number(source, "// breakpoint 2")
        lines = [breakpoint_line]
        # Set breakpoint in the thread function so we can step the threads
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )
        self.continue_to_breakpoints(breakpoint_ids)
        optimized_variable = self.dap_server.get_local_variable("argc")

        self.assertTrue(optimized_variable["value"].startswith("<error:"))
        error_msg = optimized_variable["$__lldb_extensions"]["error"]
        self.assertTrue(
            ("could not evaluate DW_OP_entry_value: no parent function" in error_msg)
            or ("variable not available" in error_msg)
        )
