"""
Test lldb-dap command hooks
"""

from lldbsuite.test.tools.lldb_dap.types import AttachArgs, LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_commands(DAPTestCaseBase):
    def test_command_directive_quiet_on_success(self):
        program = self.getBuildArtifact("a.out")
        quiet_command = (
            "settings set target.show-hex-variable-values-with-leading-zeroes false"
        )
        visible_command = (
            "settings set target.show-hex-variable-values-with-leading-zeroes true"
        )
        commands = [f"?{quiet_command}", visible_command]
        session = self.build_and_create_session()
        process_event = session.launch(
            LaunchArgs(
                program,
                initCommands=commands,
                terminateCommands=commands,
                stopCommands=commands,
                exitCommands=commands,
            )
        )
        session.verify_process_exited(after=process_event)
        full_output = session.get_console()
        self.assertNotIn(quiet_command, full_output)
        self.assertIn(visible_command, full_output)

    def do_test_abort_on_error(
        self,
        use_init_commands: bool = False,
        use_launch_commands: bool = False,
        use_pre_run_commands: bool = False,
        use_post_run_commands: bool = False,
    ):
        program = self.getBuildArtifact("a.out")
        quiet_command = (
            "settings set target.show-hex-variable-values-with-leading-zeroes false"
        )
        fake_command = "settings set foo bar"
        commands = [f"?!{quiet_command}", f"!{fake_command}"]

        session = self.build_and_create_session()
        pending_response = session.initialize_and_launch(
            LaunchArgs(
                program,
                initCommands=commands if use_init_commands else None,
                launchCommands=commands if use_launch_commands else None,
                preRunCommands=commands if use_pre_run_commands else None,
                postRunCommands=commands if use_post_run_commands else None,
            )
        )
        session.verify_configuration_done(use_post_run_commands)
        pending_response.result_or_error()
        full_output = session.get_console()
        self.assertNotIn(quiet_command, full_output)
        self.assertIn(fake_command, full_output)

    def test_command_directive_abort_on_error_init_commands(self):
        self.do_test_abort_on_error(use_init_commands=True)

    def test_command_directive_abort_on_error_launch_commands(self):
        self.do_test_abort_on_error(use_launch_commands=True)

    def test_command_directive_abort_on_error_pre_run_commands(self):
        self.do_test_abort_on_error(use_pre_run_commands=True)

    def test_command_directive_abort_on_error_post_run_commands(self):
        self.do_test_abort_on_error(use_post_run_commands=True)

    def test_command_directive_abort_on_error_attach_commands(self):
        program = self.getBuildArtifact("a.out")
        quiet_command = (
            "settings set target.show-hex-variable-values-with-leading-zeroes false"
        )
        fake_command = "settings set foo bar"
        session = self.build_and_create_session()
        session.initialize_sequence(session.initialize_args)
        attach_args = AttachArgs(
            program=program,
            attachCommands=[f"?!{quiet_command}", f"!{fake_command}"],
        )
        with self.assertRaises(AssertionError):
            session.attach(attach_args)

        full_output = session.get_console()
        self.assertNotIn(quiet_command, full_output)
        self.assertIn(fake_command, full_output)
