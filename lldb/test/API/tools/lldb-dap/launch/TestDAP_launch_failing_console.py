"""
Test lldb-dap launch request.
"""

import lldbdap_testcase


class TestDAP_launch_failing_console(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests launching in console with an invalid terminal type.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        self.create_debug_adapter()
        launch_seq = self.launch(program, console="invalid")
        response = self.dap_server.receive_response(launch_seq)
        self.assertFalse(response["success"])
        self.assertTrue(self.get_dict_value(response, ["body", "error", "showUser"]))
        self.assertRegex(
            response["body"]["error"]["format"],
            r"unexpected value, expected 'internalConsole\', 'integratedTerminal\' or 'externalTerminal\' at arguments.console",
        )
