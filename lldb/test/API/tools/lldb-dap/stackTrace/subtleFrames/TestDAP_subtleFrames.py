"""
Test lldb-dap stack trace response
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs


class TestDAP_subtleFrames(DAPTestCaseBase):
    @add_test_categories(["libc++"])
    def test_subtleFrames(self):
        """
        Internal stack frames (such as the ones used by `std::function`) are marked as "subtle".
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = "main.cpp"
        with session.configure(LaunchArgs(program)) as ctx:
            bp_line = line_number(source, "BREAK HERE")
            bps = session.resolve_source_breakpoints(source, [bp_line])

        stop_event = session.verify_stopped_on_breakpoint(bps, after=ctx.process_event)

        thread_id = self.expect_not_none(stop_event.body.threadId)
        resp = session.stack_trace(thread_id)
        frames = resp.body.stackFrames
        for f in frames:
            if "__function" in f.name:
                self.assertEqual(f.presentationHint, "subtle")
        self.assertTrue(any(f.presentationHint == "subtle" for f in frames))

        session.continue_to_exit()
