"""
Test lldb-vscode variables/stackTrace request for optimized code
"""

import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase


class TestVSCode_optimized(lldbvscode_testcase.VSCodeTestCaseBase):
    @skipIfWindows
    @skipIfRemote
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
        leaf_frame = self.vscode.get_stackFrame(frameIndex=0)
        self.assertTrue(leaf_frame["name"].endswith(" [opt]"))
        parent_frame = self.vscode.get_stackFrame(frameIndex=1)
        self.assertTrue(parent_frame["name"].endswith(" [opt]"))

    @skipIfWindows
    @skipIfRemote
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
        optimized_variable = self.vscode.get_local_variable("argc")

        self.assertTrue(optimized_variable["value"].startswith("<error:"))
