"""
Test the redirection after launching in the internal console.
"""

from lldbsuite.test.decorators import skipIfWindows
from DAP_launch_io import DAP_launchIO


@skipIfWindows
class TestDAP_launch_io_InternalConsole(DAP_launchIO):
    console = "internalConsole"
    __debuggee_stdout = None

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
        # self.get_stdout is not idempotent.
        if self.__debuggee_stdout is None:
            self.__debuggee_stdout = self.get_stdout()
        return self.__debuggee_stdout

    def _get_debuggee_stderr(self) -> str:
        # NOTE: In internalConsole stderr writes to stdout.
        return self._get_debuggee_stdout()
