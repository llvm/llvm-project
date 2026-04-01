"""
Test lldb-dap unknown request.
"""

import lldbdap_testcase


class TestDAP_unknown_request(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests handling of unknown request.
    """

    def test_no_arguments(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, stopOnEntry=True)
        self.dap_server.request_configurationDone()
        self.dap_server.wait_for_stopped()

        response = self.dap_server.request_custom("unknown")
        self.assertFalse(response["success"])
        self.assertEqual(response["body"]["error"]["format"], "unknown request")

        self.continue_to_exit()

    def test_with_arguments(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, stopOnEntry=True)
        self.dap_server.request_configurationDone()
        self.dap_server.wait_for_stopped()

        response = self.dap_server.request_custom("unknown", {"foo": "bar", "id": 42})
        self.assertFalse(response["success"])
        self.assertEqual(response["body"]["error"]["format"], "unknown request")

        self.continue_to_exit()
