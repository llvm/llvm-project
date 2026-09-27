"""
Test lldb-dap launch request.
"""

from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_launch_failing_console(DAPTestCaseBase):
    """
    Tests launching in console with an invalid terminal type.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        # No build needed: the launch request should be rejected during arg
        # validation, before lldb-dap touches the program path.
        session = self.create_session()
        session.initialize_sequence(session.initialize_args)

        err_response = session.send_request(
            LaunchArgs(program=program, console="invalid")
        ).error()
        error_msg = self.expect_not_none(
            err_response.body and err_response.body.error,
            "expected an error message in the launch response",
        )
        self.assertTrue(error_msg.showUser, "expected showUser=true")
        self.assertRegex(
            error_msg.format,
            r"unexpected value, expected 'internalConsole', 'integratedTerminal' or 'externalTerminal' at arguments.console",
        )
