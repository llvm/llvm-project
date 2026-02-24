"""
Test that all debuggee output is captured when the process exits quickly
after producing output. This exercises the ConPTY pipe drain logic on
Windows to ensure no data is lost at process exit.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_conpty_drain(lldbdap_testcase.DAPTestCaseBase):
    @skipUnlessWindows
    def test_output_not_lost_at_exit(self):
        """Test that stdout is fully captured when the debuggee exits
        immediately after writing output, exercising the ConPTY drain path."""
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, disconnectAutomatically=False)
        self.continue_to_exit()
        self.dap_server.request_disconnect()

        output = self.get_stdout()
        self.assertIsNotNone(output, "expect program stdout")
        self.assertIn(
            "DONE",
            output,
            "final output marker not found, data was lost in the ConPTY pipe: "
            + repr(output[-200:] if output else output),
        )
        # Verify we got a reasonable amount of the output
        self.assertIn("line 99:", output, "last numbered line not found")
        self.assertIn("line 0:", output, "first numbered line not found")
