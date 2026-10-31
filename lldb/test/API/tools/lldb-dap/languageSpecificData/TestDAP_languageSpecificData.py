"""
Test the language specific data lldb-dap reports for a stack frame.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs


class TestDAP_languageSpecificData(DAPTestCaseBase):
    @requireDarwin
    def test_objc_frame(self):
        """Test an Objective-C frame reports the runtime's language specific data."""
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = "main.m"
        breakpoint_line = line_number(source, "// breakpoint")

        with session.configure(LaunchArgs(program)) as ctx:
            bp_ids = session.resolve_source_breakpoints(source, [breakpoint_line])

        stop_event = session.verify_stopped_on_breakpoint(
            bp_ids, after=ctx.process_event
        )
        thread_id = self.expect_not_none(stop_event.body.threadId)
        frame = session.stack_trace(thread_id).body.stackFrames[0]
        data = self.expect_not_none(frame.languageSpecificData)
        self.assertEqual(data["Objective-C runtime version"], 2)
        session.continue_to_exit()
