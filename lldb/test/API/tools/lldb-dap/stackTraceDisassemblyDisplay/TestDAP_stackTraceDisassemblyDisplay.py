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
        other_source_file = self.getBuildArtifact("other.c")
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
        self.assertLessEqual(2, len(frames), "expect at least 2 frames")
        source_0 = frames[0].get("source")
        source_1 = frames[1].get("source")
        self.assertIsNotNone(source_0, "Expects a source object in frame 0")
        self.assertIsNotNone(source_1, "Expects a source object in frame 1")

        # it does not always have a path.
        source_0_path: str = source_0.get("path", "")
        source_1_path: str = source_1.get("path", "")

        if other_frame_assembly:
            self.assertFalse(
                source_0_path.endswith("other.c"),
                "Expect original source path to not be in unavailable source frame (other.c)",
            )
            self.assertIn(
                "sourceReference",
                source_0,
                "Expect sourceReference to be in unavailable source frame (other.c)",
            )
        else:
            self.assertTrue(
                source_0_path.endswith("other.c"),
                "Expect original source path to be in normal source frame (other.c)",
            )
            self.assertNotIn(
                "sourceReference",
                source_0,
                "Expect sourceReference to not be in normal source frame (other.c)",
            )

        if main_frame_assembly:
            self.assertFalse(
                source_1_path.endswith("main.c"),
                "Expect original source path to not be in unavailable source frame (main.c)",
            )
            self.assertIn(
                "sourceReference",
                source_1,
                "Expect sourceReference to be in unavailable source frame (main.c)",
            )
        else:
            self.assertTrue(
                source_1_path.endswith("main.c"),
                "Expect original source path to be in normal source frame (main.c)",
            )
            self.assertNotIn(
                "sourceReference",
                source_1,
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
        self.continue_to_exit()
