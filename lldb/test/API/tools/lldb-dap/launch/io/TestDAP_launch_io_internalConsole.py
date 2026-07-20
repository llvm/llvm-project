"""
Test the redirection after launching in the internal console.
"""

from DAP_launch_io import DAP_launchIO
from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.tools.lldb_dap import DAPTestSession
from lldbsuite.test.tools.lldb_dap.types import Console


@skipIfWindows
class TestDAP_launch_io_InternalConsole(DAP_launchIO):
    console = Console.INTERNAL

    def test_all_redirection(self):
        self.all_redirection(console=self.console)

    def test_stdin_redirection(self):
        self.stdin_redirection(console=self.console)

    def test_stdout_redirection(self):
        self.stdout_redirection(console=self.console)

    def test_stderr_redirection(self):
        self.stderr_redirection(console=self.console)

    def _get_debuggee_stdout(self, session: DAPTestSession) -> str:
        return session.get_stdout()

    def _get_debuggee_stderr(self, session: DAPTestSession) -> str:
        # NOTE: In internalConsole stderr writes to stdout.
        return self._get_debuggee_stdout(session)
