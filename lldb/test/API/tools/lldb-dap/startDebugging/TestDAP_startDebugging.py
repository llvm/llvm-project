"""
Test lldb-dap start-debugging reverse requests.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_startDebugging(lldbdap_testcase.DAPTestCaseBase):
    def test_startDebugging(self):
        """
        Tests the "startDebugging" reverse request. It makes sure that the IDE can
        start a child debug session.
        """
        program = self.getBuildArtifact("a.out")
        source = "main.c"
        self.build_and_launch(program)

        breakpoint_line = line_number(source, "// breakpoint")

        self.set_source_breakpoints(source, [breakpoint_line])
        self.continue_to_next_stop()
        self.dap_server.request_evaluate(
            "`lldb-dap start-debugging attach '{\"pid\":321}'", context="repl"
        )

        self.continue_to_exit()

        self.assertEqual(
            len(self.dap_server.reverse_requests),
            1,
            "make sure we got a reverse request",
        )

        request = self.dap_server.reverse_requests[0]
        self.assertEqual(request["arguments"]["configuration"]["pid"], 321)
        self.assertEqual(request["arguments"]["request"], "attach")

    def test_startDebugging_debugger_reuse(self):
        """
        Tests that debugger and target IDs can be passed through startDebugging
        for debugger reuse. This verifies the infrastructure for child DAP
        sessions to reuse the parent's debugger and attach to an existing target.
        """
        program = self.getBuildArtifact("a.out")
        source = "main.c"
        self.build_and_launch(program)

        breakpoint_line = line_number(source, "// breakpoint")
        self.set_source_breakpoints(source, [breakpoint_line])
        self.continue_to_next_stop()

        # Use mock IDs to test the infrastructure
        # In a real scenario, these would come from the parent session
        debugger_id = 1
        target_id = 100

        # Send a startDebugging request with debuggerId and targetId
        # This simulates creating a child DAP session that reuses the debugger
        session = {"session": {"debuggerId": debugger_id, "targetId": target_id}}
        self.dap_server.request_evaluate(
            f"`lldb-dap start-debugging attach '{json.dumps(session)}'",
            context="repl",
        )

        self.continue_to_exit()

        # Verify the reverse request was sent with the correct IDs
        self.assertEqual(
            len(self.dap_server.reverse_requests),
            1,
            "Should have received one startDebugging reverse request",
        )

        request = self.dap_server.reverse_requests[0]
        self.assertEqual(request["command"], "startDebugging")
        self.assertEqual(request["arguments"]["request"], "attach")

        session = request["arguments"]["configuration"]["session"]
        self.assertEqual(
            session["debuggerId"],
            debugger_id,
            "Reverse request should include debugger ID",
        )
        self.assertEqual(
            session["targetId"],
            target_id,
            "Reverse request should include target ID",
        )
