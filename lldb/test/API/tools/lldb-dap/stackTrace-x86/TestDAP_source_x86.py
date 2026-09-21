"""
Test lldb-dap stack trace containing x86 assembly
"""

from lldbsuite.test import lldbplatformutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs


class TestDAP_stacktrace_x86(DAPTestCaseBase):
    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"] + lldbplatformutil.getDarwinOSTriples())
    def test_stacktrace_x86(self):
        """Tests that lldb-dap steps through x86 assembly correctly and reports the right source lines."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        launch_args = LaunchArgs(
            program,
            initCommands=[
                "settings set target.process.thread.step-in-avoid-nodebug false"
            ],
        )
        with session.configure(launch_args) as ctx:
            source = "main.c"
            [breakpoint_id] = session.resolve_source_breakpoints(
                source, [line_number(source, "// Break here")]
            )

        stop_event = session.verify_stopped_on_breakpoint(
            breakpoint_id, after=ctx.process_event
        )
        thread_ctx = session.thread_context_from(stop_event)
        thread_ctx.step_in()

        frame = thread_ctx.top_frame().frame
        self.assertEqual(
            frame.name,
            "no_branch_func",
            "expected to be in the no_branch_func function",
        )
        self.assertEqual(frame.line, 1, "expected to be at the start of the function")

        minimum_assembly_lines = (
            line_number(source, "Assembly end")
            - line_number(source, "Assembly start")
            + 1
        )
        self.assertGreaterEqual(
            minimum_assembly_lines,
            10,
            "expected a reasonable number of assembly lines",
        )

        for expected_line in range(2, minimum_assembly_lines):
            thread_ctx.step_in()
            top_frame = thread_ctx.top_frame().frame
            self.assertEqual(
                top_frame.name,
                "no_branch_func",
                "expected to still be in the no_branch_func function",
            )
            self.assertEqual(
                top_frame.line,
                expected_line,
                f"step-in should advance a single line in the function to {expected_line}",
            )
