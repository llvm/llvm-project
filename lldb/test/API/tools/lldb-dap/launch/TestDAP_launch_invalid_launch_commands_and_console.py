"""
Test lldb-dap launch request.
"""

from lldbsuite.test.tools.lldb_dap.types import Console, LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_launch_invalid_launch_commands_and_console(DAPTestCaseBase):
    """
    Tests launching with launch commands in an integrated terminal.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        # No build needed: the launch request should be rejected during
        # argument validation, before lldb-dap touches the program path.
        session = self.create_session()
        session.initialize_sequence(session.initialize_args)

        err_response = session.send_request(
            LaunchArgs(
                program=program,
                launchCommands=["a b c"],
                console=Console.INTEGRATED_TERMINAL,
            )
        ).error()
        error_msg = self.expect_not_none(
            err_response.body and err_response.body.error,
            "expected an error message in the launch response",
        )
        self.assertTrue(error_msg.showUser, "expected showUser=true")
        self.assertIn(
            "'launchCommands' and non-internal 'console' are mutually exclusive",
            error_msg.format,
        )
