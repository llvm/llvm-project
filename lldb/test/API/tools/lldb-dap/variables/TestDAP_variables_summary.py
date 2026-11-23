"""
Test variables with summary.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_variables_summary(DAPTestCaseBase):
    def test_summary_no_value(self):
        """A variable whose summary formatter hides the value should only show
        the summary. similar to the command `frame variable`."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()

        int_summary = "fake_summary"
        launch_args = LaunchArgs(
            program=program,
            initCommands=[
                f'type summary add --no-value --summary-string "{int_summary}" int'
            ],
        )
        with session.configure(launch_args) as ctx:
            source = "main.cpp"
            bp_line = line_number(source, "// breakpoint 1")
            [bp] = session.resolve_source_breakpoints(source, [bp_line])

        stop_event = session.verify_stopped_on_breakpoint(bp, after=ctx.process_event)
        frame = session.top_frame_from(stop_event)

        self.assertEqual(frame.locals["argc"].value, int_summary)
        session.verify_evaluate(frame.evaluate("argc"), type="int", result=int_summary)

        set_resp = self.expect_success(frame.locals.set("argc", 20))
        self.assertEqual(set_resp.body.type, "int")
        self.assertEqual(set_resp.body.value, int_summary)

        # The string summary of a `const char *` doesn't hide its value.
        valid_str = frame.locals["valid_str"].value
        self.assertRegex(valid_str, r'^0x[0-9a-f]+ "')

        session.continue_to_exit()
