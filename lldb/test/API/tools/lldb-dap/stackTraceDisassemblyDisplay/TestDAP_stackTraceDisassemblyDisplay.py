"""
Test lldb-dap stack trace when some of the source paths are missing
"""

from lldbsuite.test.lldbtest import line_number
import lldbdap_testcase
from contextlib import contextmanager
import os


OTHER_C_SOURCE_CODE = """
int no_source_func(int n) {
    return n + 1; // Break here
}
"""


@contextmanager
def delete_file_on_exit(path):
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)


class TestDAP_stackTraceMissingSourcePath(lldbdap_testcase.DAPTestCaseBase):
    def build_and_run_until_breakpoint(self):
        """
        Build the program and run until the breakpoint is hit, and return the stack frames.
        """
        other_source_file = "other.c"
        with delete_file_on_exit(other_source_file):
            with open(other_source_file, "w") as f:
                f.write(OTHER_C_SOURCE_CODE)

            breakpoint_line = line_number(other_source_file, "// Break here")

            program = self.getBuildArtifact("a.out")
            self.build_and_launch(program, commandEscapePrefix="")

            breakpoint_ids = self.set_source_breakpoints(
                other_source_file, [breakpoint_line]
            )
            self.assertEqual(
                len(breakpoint_ids), 1, "expect correct number of breakpoints"
            )

            self.continue_to_breakpoints(breakpoint_ids)

        frames = self.get_stackFrames()
        self.assertLessEqual(2, len(frames), "expect at least 2 frames")

        self.assertIn(
            "path",
            frames[0]["source"],
            "Expect source path to always be in frame (other.c)",
        )
        self.assertIn(
            "path",
            frames[1]["source"],
            "Expect source path in always be in frame (main.c)",
        )

        return frames

    def verify_frames_source(
        self, frames, main_frame_assembly: bool, other_frame_assembly: bool
    ):
        if other_frame_assembly:
            self.assertFalse(
                frames[0]["source"]["path"].endswith("other.c"),
                "Expect original source path to not be in unavailable source frame (other.c)",
            )
            self.assertIn(
                "sourceReference",
                frames[0]["source"],
                "Expect sourceReference to be in unavailable source frame (other.c)",
            )
        else:
            self.assertTrue(
                frames[0]["source"]["path"].endswith("other.c"),
                "Expect original source path to be in normal source frame (other.c)",
            )
            self.assertNotIn(
                "sourceReference",
                frames[0]["source"],
                "Expect sourceReference to not be in normal source frame (other.c)",
            )

        if main_frame_assembly:
            self.assertFalse(
                frames[1]["source"]["path"].endswith("main.c"),
                "Expect original source path to not be in unavailable source frame (main.c)",
            )
            self.assertIn(
                "sourceReference",
                frames[1]["source"],
                "Expect sourceReference to be in unavailable source frame (main.c)",
            )
        else:
            self.assertTrue(
                frames[1]["source"]["path"].endswith("main.c"),
                "Expect original source path to be in normal source frame (main.c)",
            )
            self.assertNotIn(
                "sourceReference",
                frames[1]["source"],
                "Expect sourceReference to not be in normal source code frame (main.c)",
            )

    def test_stopDisassemblyDisplay(self):
        """
        Test that with with all stop-disassembly-display values you get correct assembly / no assembly source code.
        """
        self.build_and_run_until_breakpoint()
        frames = self.get_stackFrames()
        self.assertLessEqual(2, len(frames), "expect at least 2 frames")

        self.assertIn(
            "path",
            frames[0]["source"],
            "Expect source path to always be in frame (other.c)",
        )
        self.assertIn(
            "path",
            frames[1]["source"],
            "Expect source path in always be in frame (main.c)",
        )

        self.dap_server.request_evaluate(
            "settings set stop-disassembly-display never", context="repl"
        )
        frames = self.get_stackFrames()
        self.verify_frames_source(
            frames, main_frame_assembly=False, other_frame_assembly=False
        )

        self.dap_server.request_evaluate(
            "settings set stop-disassembly-display always", context="repl"
        )
        frames = self.get_stackFrames()
        self.verify_frames_source(
            frames, main_frame_assembly=True, other_frame_assembly=True
        )

        self.dap_server.request_evaluate(
            "settings set stop-disassembly-display no-source", context="repl"
        )
        frames = self.get_stackFrames()
        self.verify_frames_source(
            frames, main_frame_assembly=False, other_frame_assembly=True
        )

        self.dap_server.request_evaluate(
            "settings set stop-disassembly-display no-debuginfo", context="repl"
        )
        frames = self.get_stackFrames()
        self.verify_frames_source(
            frames, main_frame_assembly=False, other_frame_assembly=False
        )
