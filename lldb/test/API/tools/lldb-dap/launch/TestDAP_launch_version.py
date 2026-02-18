"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import expectedFailureWindows
from lldbsuite.test.lldbtest import line_number
import lldbdap_testcase


class TestDAP_launch_version(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests that "initialize" response contains the "version" string the same
    as the one returned by "version" command.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        source = "main.c"
        breakpoint_line = line_number(source, "// breakpoint 1")
        lines = [breakpoint_line]
        # Set breakpoint in the thread function so we can step the threads
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.continue_to_breakpoints(breakpoint_ids)

        version_eval_response = self.dap_server.request_evaluate(
            "`version", context="repl"
        )
        version_eval_output = version_eval_response["body"]["result"]

        version_string = self.dap_server.get_capability("$__lldb_version")
        self.assertEqual(
            version_eval_output.splitlines(),
            version_string.splitlines(),
            "version string does not match",
        )
