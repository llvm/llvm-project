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
