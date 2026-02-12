"""
Test lldb-dap custom request.
"""

from lldbsuite.test.decorators import expectedFailureWindows
import lldbdap_testcase


class TestDAP_unknown_request(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests handling of unknown request.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, stopOnEntry=True)
        self.dap_server.request_configurationDone()
        self.dap_server.wait_for_stopped()

        response = self.dap_server.request_custom("unknown")
        self.assertFalse(response["success"])
        self.assertEqual(response["body"]["error"]["format"], "Unknown request")

        self.continue_to_exit()