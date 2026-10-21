"""
Test lldb-dap start-debugging reverse requests.
"""

import json
from typing import cast

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import *


class TestDAP_startDebugging(DAPTestCaseBase):
    def test_startDebugging(self):
        """
        Tests the "startDebugging" reverse request. It makes sure that the IDE can
        start a child debug session.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        with session.configure(LaunchArgs(program)) as ctx:
            source = "main.c"
            breakpoint_line = line_number(source, "// breakpoint")
            [bp_id] = session.resolve_source_breakpoints(source, [breakpoint_line])

        session.verify_stopped_on_breakpoint(bp_id, after=ctx.process_event)
        session.evaluate(
            "`lldb-dap start-debugging attach '{\"pid\":321}'", context="repl"
        )
        session.continue_to_exit()

        args = self.start_debugging_args(session)
        self.assertIsInstance(args.configuration, AttachArgs)
        configuration = cast(AttachArgs, args.configuration)
        self.assertEqual(configuration.pid, 321)
        self.assertEqual(args.request, StartDebuggingRequestType.ATTACH)

    def test_startDebugging_debugger_reuse(self):
        """
        Tests that debugger and target IDs can be passed through startDebugging
        for debugger reuse. This verifies the infrastructure for child DAP
        sessions to reuse the parent's debugger and attach to an existing target.
        """
        program = self.getBuildArtifact("a.out")
        source = "main.c"
        session = self.build_and_create_session()

        with session.configure(LaunchArgs(program)) as ctx:
            breakpoint_line = line_number(source, "// breakpoint")
            [bp_id] = session.resolve_source_breakpoints(source, [breakpoint_line])
        session.verify_stopped_on_breakpoint(bp_id, after=ctx.process_event)

        # Use mock IDs to test the infrastructure.
        # In a real scenario, these would come from the parent session.
        debugger_id = 1
        target_id = 100

        # Send a startDebugging request with debuggerId and targetId
        # This simulates creating a child DAP session that reuses the debugger
        attach_args = {"session": {"debuggerId": debugger_id, "targetId": target_id}}
        session.evaluate(
            f"`lldb-dap start-debugging attach '{json.dumps(attach_args)}'",
            context="repl",
        )
        session.continue_to_exit()

        args = self.start_debugging_args(session)
        self.assertEqual(args.request, StartDebuggingRequestType.ATTACH)
        self.assertIsInstance(args.configuration, AttachArgs)
        configuration = cast(AttachArgs, args.configuration)

        child_session = self.expect_not_none(configuration.session)
        self.assertEqual(
            child_session.debuggerId,
            debugger_id,
            "Reverse request should include debugger ID",
        )
        self.assertEqual(
            child_session.targetId,
            target_id,
            "Reverse request should include target ID",
        )

    def start_debugging_args(self, session) -> StartDebuggingRequestArgs:
        request = session.last_reverse_request()
        self.assertIsInstance(request, StartDebuggingRequest)
        self.assertEqual(request.command, "startDebugging")
        return cast(StartDebuggingRequest, request).arguments
