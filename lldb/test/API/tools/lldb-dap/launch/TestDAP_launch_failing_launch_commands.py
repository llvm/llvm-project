"""
Test lldb-dap launch request.
"""

import os
import re

from lldbsuite.test.tools.lldb_dap.dap_types import InitializedEvent, LaunchArgs
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase


class TestDAP_launch_failing_launch_commands(DAPTestCaseBase):
    """
    Tests "launchCommands" failures prevents a launch.
    """

    def test(self):
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")
        init_response = session.initialize_sequence(session.initialize_args)

        # Run an invalid launch command, in this case a bad path.
        bad_path = os.path.join("bad", "path")
        launchCommands = [f'!target create "{bad_path}{program}"']
        initCommands = ["target list", "platform list"]
        preRunCommands = ["image list a.out", "image dump sections a.out"]

        launch_handle = session.send_request(
            LaunchArgs(
                program=program,
                initCommands=initCommands,
                preRunCommands=preRunCommands,
                launchCommands=launchCommands,
            )
        )
        session.wait_for_event(InitializedEvent, after=init_response)
        session.configuration_done().result_or_error()
        err_response = launch_handle.error()

        error_msg = self.expect_not_none(
            err_response.body and err_response.body.error,
            "expected an error message in the launch response",
        )
        self.assertRegex(
            error_msg.format,
            r"Failed to run launch commands\. See the Debug Console for more details",
        )

        # Get output from the console. This should contain the
        # "initCommands", "preRunCommands", and "launchCommands".
        output = session.get_console()
        session.verify_commands("initCommands", output, initCommands)
        session.verify_commands("preRunCommands", output, preRunCommands)
        session.verify_commands("launchCommands", output, launchCommands)
        # The launch should fail due to the invalid command.
        self.assertRegex(output, re.escape(bad_path) + r".*does not exist")
