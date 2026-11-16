"""
Test lldb-dap launch request.
"""

from lldbsuite.test.tools.lldb_dap.types import InitializedEvent, LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_launch_invalid_program(DAPTestCaseBase):
    """
    Tests launching with an invalid program.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        session = self.create_session()
        session.initialize_sequence(session.initialize_args)
        launch_handle = session.send_request(LaunchArgs(program=program))

        err_response = launch_handle.error()
        session.verify_configuration_done(expected_success=False)
        error_msg = self.expect_not_none(
            err_response.body and err_response.body.error,
            "expected an error message in the launch response",
        )
        self.assertEqual(error_msg.format, f"'{program}' does not exist")

        # Verify we never received an initialized event.
        with self.assertRaises(TimeoutError):
            session.wait_for_earliest_event(InitializedEvent, timeout=0)
