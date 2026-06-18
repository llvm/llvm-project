"""
Test lldb-dap launch request.
"""

import lldbdap_testcase


class TestDAP_launch_invalid_program(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests launching with an invalid program.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        self.create_debug_adapter()
        launch_seq = self.launch(program)
        self.dap_server.wait_for_initialized()
        self.dap_server.request_configurationDone()
        response = self.dap_server.receive_response(launch_seq)
        self.assertFalse(response["success"])
        self.assertEqual(
            "'{0}' does not exist".format(program), response["body"]["error"]["format"]
        )
