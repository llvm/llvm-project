"""
Test lldb-dap repl mode detection
"""

from typing import Optional

from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_repl_mode_detection(DAPTestCaseBase):
    def assertEvaluate(
        self, expression: str, regex: str, frame_id: Optional[int] = None
    ):
        result = self._session.evaluate(
            expression, context="repl", frameId=frame_id
        ).result
        self.assertRegex(result, regex)

    def test_completions(self):
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        self._session = session

        source = "main.cpp"
        breakpoint1_line = line_number(source, "// breakpoint 1")
        breakpoint2_line = line_number(source, "// breakpoint 2")

        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(
                source, [breakpoint1_line, breakpoint2_line]
            )

            self.assertEvaluate("lldb-dap repl-mode", "auto")
            # The result of the commands should return the empty string.
            self.assertEvaluate("`command regex user_command s/^$/platform/", r"^$")
            self.assertEvaluate("`command alias alias_command platform", r"^$")
            self.assertEvaluate(
                "`command alias alias_command_with_arg platform select --sysroot %1 remote-linux",
                r"^$",
            )

        # Stop in `fun`. Locals shadow the command names, so evaluating
        # the identifiers should return the local integer values.
        stop_event = session.verify_stopped_on_breakpoint(after=ctx.process_event)
        top_frame_id = session.top_frame_from(stop_event).id

        self.assertEvaluate("user_command", "474747", top_frame_id)
        self.assertEvaluate("alias_command", "474747", top_frame_id)
        self.assertEvaluate("alias_command_with_arg", "474747", top_frame_id)
        self.assertEvaluate("platform", "474747", top_frame_id)

        # Stop back in `main`. With no shadowing locals, the same names
        # should resolve to their lldb command counterparts.
        stop_event = session.continue_to_next_stop()
        top_frame_id = session.top_frame_from(stop_event).id
        platform_help_needle = "Commands to manage and create platforms"

        self.assertEvaluate("user_command", platform_help_needle, top_frame_id)
        self.assertEvaluate("alias_command", platform_help_needle, top_frame_id)
        self.assertEvaluate(
            "alias_command_with_arg " + self.getBuildDir(),
            "Platform: remote-linux",
            top_frame_id,
        )
        self.assertEvaluate("platform", platform_help_needle, top_frame_id)
