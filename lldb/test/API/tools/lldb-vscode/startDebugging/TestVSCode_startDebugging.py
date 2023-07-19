"""
Test lldb-vscode startDebugging reverse request
"""


import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase


class TestVSCode_startDebugging(lldbvscode_testcase.VSCodeTestCaseBase):
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
        self.vscode.request_evaluate(
            "`lldb-vscode startDebugging attach '{\"pid\":321}'", context="repl"
        )

        self.continue_to_exit()

        self.assertEqual(
            len(self.vscode.reverse_requests), 1, "make sure we got a reverse request"
        )

        request = self.vscode.reverse_requests[0]
        self.assertEqual(request["arguments"]["configuration"]["pid"], 321)
        self.assertEqual(request["arguments"]["request"], "attach")
