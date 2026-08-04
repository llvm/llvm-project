"""
Test the redirection after launching in the integrated terminal.
"""

from DAP_launch_io import DAP_launchIO
from lldbsuite.test.decorators import (
    skipIfAsan,
    skipIfBuildType,
    skipIfRemote,
    skipIfWindows,
)
from lldbsuite.test.tools.lldb_dap import DAPTestSession
from lldbsuite.test.tools.lldb_dap.types import Console, RunInTerminalRequest


@skipIfRemote
@skipIfAsan
@skipIfBuildType(["debug"])
@skipIfWindows
class TestDAP_launch_io_IntegratedTerminal(DAP_launchIO):
    console = Console.INTEGRATED_TERMINAL

    def test_all_redirection(self):
        self.all_redirection(console=self.console)

    def test_stdin_redirection(self):
        self.stdin_redirection(console=self.console)

    def test_stdout_redirection(self):
        self.stdout_redirection(console=self.console)

    def test_stderr_redirection(self):
        self.stderr_redirection(console=self.console)

    def _get_debuggee_stdout(self, session: DAPTestSession) -> str:
        self.assertIsInstance(session.last_reverse_request(), RunInTerminalRequest)
        return session.get_stdout()

    def _get_debuggee_stderr(self, session: DAPTestSession) -> str:
        self.assertIsInstance(session.last_reverse_request(), RunInTerminalRequest)
        return session.get_stderr()
