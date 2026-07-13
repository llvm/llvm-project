"""
Test lldb-dap launch request.
"""

from lldbsuite.test.tools.lldb_dap.dap_types import LaunchArgs
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase


class TestDAP_launch_invalid_program(DAPTestCaseBase):
    """
    Tests launching with an invalid program.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        session = self.create_session()
        session.initialize_sequence(session.initialize_args)
        launch_handle = session.send_request(LaunchArgs(program=program))
        session.ensure_initialized()
        session.verify_configuration_done(expected_success=False)

        err_response = launch_handle.error()
        error_msg = self.expect_not_none(
            err_response.body and err_response.body.error,
            "expected an error message in the launch response",
        )
        self.assertEqual(error_msg.format, f"'{program}' does not exist")
