"""
Test lldb-dap stackTrace request for compiler generated code
"""

import os

import lldbdap_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestDAP_stackTraceCompilerGeneratedCode(lldbdap_testcase.DAPTestCaseBase):
    def test_non_leaf_frame_compiler_generate_code(self):
        """
        Test that non-leaf frames with compiler-generated code are properly resolved.

        This test verifies that LLDB correctly handles stack frames containing
        compiler-generated code (code without valid source location information).
        When a non-leaf frame contains compiler-generated code immediately after a
        call instruction, LLDB should resolve the frame's source location to the
        call instruction's line, rather than to the compiler-generated code that
        follows, which lacks proper symbolication information.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.c"

        # Set breakpoint inside bar() function
        lines = [line_number(source, "// breakpoint here")]
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )

        self.continue_to_breakpoints(breakpoint_ids)

        # Get the stack frames: [0] = bar(), [1] = foo(), [2] = main()
        stack_frames = self.get_stackFrames()
        self.assertGreater(len(stack_frames), 2, "Expected more than 2 stack frames")

        # Examine the foo() frame (stack_frames[1])
        # This is the critical frame containing compiler-generated code
        foo_frame = stack_frames[1]

        # Verify that the frame's line number points to the bar() call,
        # not to the compiler-generated code after it
        foo_call_bar_source_line = foo_frame.get("line")
        self.assertEqual(
            foo_call_bar_source_line,
            line_number(source, "foo call bar"),
            "Expected foo call bar to be the source line of the frame",
        )

        # Verify the source file name is correctly resolved
        foo_source_name = foo_frame.get("source", {}).get("name")
        self.assertEqual(
            foo_source_name, "main.c", "Expected foo source name to be main.c"
        )

        # When lldb fails to symbolicate a frame it will emit a fake assembly
        # source with path of format <module>`<symbol> or <module>`<address> with
        # sourceReference to retrieve disassembly source file.
        # Verify that this didn't happen - the path should be a real file path.
        foo_path = foo_frame.get("source", {}).get("path")
        self.assertNotIn("`", foo_path, "Expected foo source path to not contain `")
        self.continue_to_exit()
