"""
Test lldb-dap stack trace when some of the source paths are missing,
under each `stop-disassembly-display` setting.
"""

import os

from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.session_helpers import ThreadContext

OTHER_C_SOURCE_CODE = """
int no_source_func(int n) {
    return n + 1; // Break here
}
"""


class TestDAP_stackTraceDisassemblyDisplay(DAPTestCaseBase):
    def build_and_run_until_other_c_breakpoint(self):
        """Build, launch, stop at the breakpoint in other.c, and return the
        thread context for the stopped thread.
        """
        other_source_file = self.getBuildArtifact("other.c")
        with open(other_source_file, "w") as f:
            f.write(OTHER_C_SOURCE_CODE)
        breakpoint_line = line_number(other_source_file, "// Break here")

        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()

        with session.configure(LaunchArgs(program, commandEscapePrefix="")) as ctx:
            breakpoint_ids = session.resolve_source_breakpoints(
                other_source_file, [breakpoint_line]
            )

        stop_event = session.verify_stopped_on_breakpoint(
            breakpoint_ids, after=ctx.process_event
        )
        # Remove other.c so LLDB sees it as "source unavailable" for the
        # subsequent stop-disassembly-display checks.
        os.remove(other_source_file)

        thread_ctx = session.thread_context_from(stop_event)

        frames = [f.frame for f in thread_ctx.frames()]
        self.assertLessEqual(2, len(frames), "expect at least 2 frames")

        frame_0_source = self.expect_not_none(frames[0].source)
        frame_1_source = self.expect_not_none(frames[1].source)
        self.assertIsNotNone(
            frame_0_source.path, "expect source path to always be in frame (other.c)"
        )
        self.assertIsNotNone(
            frame_1_source.path, "expect source path to always be in frame (main.c)"
        )
        return session, thread_ctx

    def verify_frames_source(
        self,
        thread_ctx: ThreadContext,
        main_frame_assembly: bool,
        other_frame_assembly: bool,
    ):
        frames = [f.frame for f in thread_ctx.frames(levels=2)]
        self.assertLessEqual(2, len(frames), "expect at least 2 frames")
        source_0 = self.expect_not_none(
            frames[0].source, "expects a source object in frame 0"
        )
        source_1 = self.expect_not_none(
            frames[1].source, "expects a source object in frame 1"
        )

        # It does not always have a path.
        source_0_path = source_0.path or ""
        source_1_path = source_1.path or ""

        if other_frame_assembly:
            self.assertFalse(
                source_0_path.endswith("other.c"),
                "expect original source path to not be in unavailable source frame (other.c).",
            )
            self.assertIsNotNone(
                source_0.sourceReference,
                "expect sourceReference to be in unavailable source frame (other.c).",
            )
        else:
            self.assertTrue(
                source_0_path.endswith("other.c"),
                "expect original source path to be in normal source frame (other.c).",
            )
            self.assertIsNone(
                source_0.sourceReference,
                "expect sourceReference to not be in normal source frame (other.c).",
            )

        if main_frame_assembly:
            self.assertFalse(
                source_1_path.endswith("main.c"),
                "expect original source path to not be in unavailable source frame (main.c).",
            )
            self.assertIsNotNone(
                source_1.sourceReference,
                "expect sourceReference to be in unavailable source frame (main.c).",
            )
        else:
            self.assertTrue(
                source_1_path.endswith("main.c"),
                "expect original source path to be in normal source frame (main.c).",
            )
            self.assertIsNone(
                source_1.sourceReference,
                "expect sourceReference to not be in normal source code frame (main.c).",
            )

    def test_stopDisassemblyDisplay(self):
        """With each `stop-disassembly-display` setting, the corresponding frames
        carry either an original source path or a sourceReference for assembly."""
        session, thread_ctx = self.build_and_run_until_other_c_breakpoint()

        # Baseline: both source files are present on disk.
        baseline_frames = [f.frame for f in thread_ctx.frames(levels=2)]
        self.assertLessEqual(2, len(baseline_frames), "expect at least 2 frames")
        self.assertIsNotNone(
            baseline_frames[0].source and baseline_frames[0].source.path,
            "expect source path to always be in frame (other.c)",
        )
        self.assertIsNotNone(
            baseline_frames[1].source and baseline_frames[1].source.path,
            "expect source path to always be in frame (main.c)",
        )

        session.evaluate("settings set stop-disassembly-display never", context="repl")
        self.verify_frames_source(
            thread_ctx, main_frame_assembly=False, other_frame_assembly=False
        )

        session.evaluate("settings set stop-disassembly-display always", context="repl")
        self.verify_frames_source(
            thread_ctx, main_frame_assembly=True, other_frame_assembly=True
        )

        session.evaluate(
            "settings set stop-disassembly-display no-source", context="repl"
        )
        self.verify_frames_source(
            thread_ctx, main_frame_assembly=False, other_frame_assembly=True
        )

        session.evaluate(
            "settings set stop-disassembly-display no-debuginfo", context="repl"
        )
        self.verify_frames_source(
            thread_ctx, main_frame_assembly=False, other_frame_assembly=False
        )

        session.continue_to_exit()
