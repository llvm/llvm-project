"""
Test the redirection after launching in the integrated terminal.
"""

from typing import IO
from lldbsuite.test.decorators import (
    skipIfAsan,
    skipIfBuildType,
    skipIfRemote,
    skipIfWindows,
)

from DAP_launch_io import DAP_launchIO


@skipIfRemote
@skipIfAsan
@skipIfBuildType(["debug"])
@skipIfWindows
class TestDAP_launch_io_IntegratedTerminal(DAP_launchIO):
    console = "integratedTerminal"

    # all redirection
    def test_all_redirection(self):
        self.all_redirection(console=self.console)

    def test_all_redirection_with_args(self):
        self.all_redirection(console=self.console, with_args=True)

    # stdin
    def test_stdin_redirection(self):
        self.stdin_redirection(console=self.console)

    def test_stdin_redirection_with_args(self):
        self.stdin_redirection(console=self.console, with_args=True)

    # stdout
    def test_stdout_redirection(self):
        self.stdout_redirection(console=self.console)

    def test_stdout_redirection_with_env(self):
        self.stdout_redirection(console=self.console, with_env=True)

    # stderr
    def test_stderr_redirection(self):
        self.stderr_redirection(console=self.console)

    def test_stderr_redirection_with_env(self):
        self.stderr_redirection(console=self.console, with_env=True)

    def _get_debuggee_stdout(self) -> str:
        self.assertIsNotNone(
            self.dap_server.reverse_process, "Expected a debuggee process."
        )
        proc_stdout: IO = self.dap_server.reverse_process.stdout
        self.assertIsNotNone(proc_stdout)
        return proc_stdout.read().decode()

    def _get_debuggee_stderr(self) -> str:
        self.assertIsNotNone(
            self.dap_server.reverse_process, "Expected a debuggee process."
        )
        proc_stderr = self.dap_server.reverse_process.stderr
        self.assertIsNotNone(proc_stderr)
        return proc_stderr.read().decode()
