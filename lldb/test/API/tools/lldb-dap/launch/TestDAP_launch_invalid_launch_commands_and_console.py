"""
Test lldb-dap launch request.
"""

import lldbdap_testcase


class TestDAP_launch_failing_launch_commands_and_console(
    lldbdap_testcase.DAPTestCaseBase
):
    """
    Tests launching with launch commands in an integrated terminal.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        self.create_debug_adapter()
        launch_seq = self.launch(
            program, launchCommands=["a b c"], console="integratedTerminal"
        )
        response = self.dap_server.receive_response(launch_seq)
        self.assertFalse(response["success"])
        self.assertTrue(self.get_dict_value(response, ["body", "error", "showUser"]))
        self.assertIn(
            "'launchCommands' and non-internal 'console' are mutually exclusive",
            self.get_dict_value(response, ["body", "error", "format"]),
        )
