"""
Test lldb-dap output events
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_output(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    def test_output(self):
        """
        Test output handling for the running process.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(
            program,
            disconnectAutomatically=False,
            exitCommands=[
                # Ensure that output produced by lldb itself is not consumed by the OutputRedirector.
                "?script print('out\\0\\0', end='\\r\\n', file=sys.stdout)",
                "?script print('err\\0\\0', end='\\r\\n', file=sys.stderr)",
            ],
        )
        source = "main.c"
        lines = [line_number(source, "// breakpoint 1")]
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.continue_to_breakpoints(breakpoint_ids)

        # Ensure partial messages are still sent.
        output = self.collect_stdout(pattern="abcdef")
        self.assertTrue(output and len(output) > 0, "expect program stdout")

        self.continue_to_exit()

        # Disconnecting from the server to ensure any pending IO is flushed.
        self.dap_server.request_disconnect()

        output += self.get_stdout()
        self.assertTrue(output and len(output) > 0, "expect program stdout")
        self.assertIn(
            "abcdefghi\r\nhello world\r\nfinally\0\0",
            output,
            "full stdout not found in: " + repr(output),
        )
        console = self.get_console()
        self.assertTrue(console and len(console) > 0, "expect dap messages")
        self.assertIn(
            "out\0\0\r\nerr\0\0\r\n", console, f"full console message not found"
        )
