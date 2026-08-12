"""
Test lldb-dap source request
"""

from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs, Source, SourceArgs


class TestDAP_source(DAPTestCaseBase):
    @skipIfWindows
    def test_source(self):
        """Tests the Source Request."""
        program = self.getBuildArtifact("a.out")
        source = self.getSourcePath("main.c")
        session = self.build_and_create_session()
        with session.configure(LaunchArgs(program)) as ctx:
            breakpoint_line = line_number(source, "breakpoint")
            [bp_id] = session.resolve_source_breakpoints(source, [breakpoint_line])

        stop_event = session.verify_stopped_on_breakpoint(
            bp_id, after=ctx.process_event
        )

        src_args = SourceArgs(source=Source(sourceReference=0), sourceReference=0)
        session.send_request(src_args).error()
        # Check only source reference in the arguments field.
        resp = session.send_request(SourceArgs(sourceReference=0)).error(
            "verify invalid sourceReference fails"
        )
        resp_body = self.expect_not_none(resp.body)
        error_msg = self.expect_not_none(resp_body.error)
        self.assertIn("unknown source reference", error_msg.format)

        # Verify the top three frames handler, add and main.
        thread_id = session.thread_context_from(stop_event).thread_id
        response = session.stack_trace(thread_id)
        frames = response.body.stackFrames
        self.assertGreaterEqual(len(frames), 3, "verify we got up to main at least.")
        self.assertEqual(
            len(frames),
            response.body.totalFrames,
            "verify total frames returns a speculative page size",
        )

        handler_frame, add_frame, main_frame, *_ = frames

        # Verify frame 0 handler.
        self.assertEqual(handler_frame.name, "handler")
        self.assertEqual(handler_frame.line, line_number(source, "first_frame"))
        handler_source = self.expect_not_none(handler_frame.source)
        self.assertEqual(handler_source.name, "main.c")
        self.assertEqual(handler_source.path, source)
        self.assertIsNone(handler_source.sourceReference)

        # Verify frame 1 add.
        self.assertEqual(add_frame.name, "add")
        add_source = self.expect_not_none(add_frame.source)
        self.assertEqual(add_source.name, "add")
        self.assertEqual(add_source.path, program + "`add")

        source_ref = self.expect_not_none(add_source.sourceReference)
        disasm = session.send_request(SourceArgs(source_ref)).result()
        self.assertGreater(
            len(disasm.body.content), 0, "verify content returned disassembly"
        )
        self.assertEqual(
            disasm.body.mimeType, "text/x-lldb.disassembly", "verify mime type returned"
        )

        # Verify frame 2 main.
        self.assertEqual(main_frame.name, "main")
        self.assertEqual(main_frame.line, line_number(source, "third_frame"))
        main_source = self.expect_not_none(main_frame.source)
        self.assertEqual(main_source.name, "main.c")
        self.assertEqual(main_source.path, source)
        self.assertIsNone(main_source.sourceReference)

        session.continue_to_exit()
