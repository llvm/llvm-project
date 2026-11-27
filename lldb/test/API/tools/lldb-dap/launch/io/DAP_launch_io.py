"""
Test the redirection of stdio.
There are three ways to launch the debuggee:
internalConsole, integratedTerminal and externalTerminal.

For each redirection configuration we check the stdin, argv, and env
input paths. The C++ test program writes whatever it receives from
each available source.

NOTE: The testcases do not include all possible configurations of
consoles and input sources.
"""

from abc import abstractmethod
from tempfile import NamedTemporaryFile

from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase, DAPTestSession
from lldbsuite.test.tools.lldb_dap.types import Console, LaunchArgs


class DAP_launchIO(DAPTestCaseBase):
    """Implements the redirection scenarios that are common to every console.

    Subclasses provide `console` and override `_get_debuggee_stdout` /
    `_get_debuggee_stderr` for the cases where stdout / stderr are not
    redirected to files (the streams have to be read from the console
    instead, which differs between InternalConsole and IntegratedTerminal).
    """

    def all_redirection(self, console: Console):
        """All three streams redirected to files. Verify every input path."""
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")
        stdin_input = "from stdin"
        args_input = "from argv"
        env_input = "from env"

        with NamedTemporaryFile("wt") as stdin, NamedTemporaryFile(
            "rt"
        ) as stdout, NamedTemporaryFile("rt") as stderr:
            stdin.write(stdin_input)
            stdin.flush()

            session.launch(
                LaunchArgs(
                    program,
                    stdio=[stdin.name, stdout.name, stderr.name],
                    console=console,
                    args=["--read-stdin", args_input],
                    env={"FROM_ENV": env_input},
                )
            )
            session.verify_process_exited()

            stdout_text = stdout.read()
            stderr_text = stderr.read()
            self.assertIn(f"[STDOUT][FROM_STDIN]: {stdin_input}", stdout_text)
            self.assertIn(f"[STDOUT][FROM_ARGV]: {args_input}", stdout_text)
            self.assertIn(f"[STDOUT][FROM_ENV]: {env_input}", stdout_text)

            self.assertIn(f"[STDERR][FROM_STDIN]: {stdin_input}", stderr_text)
            self.assertIn(f"[STDERR][FROM_ARGV]: {args_input}", stderr_text)
            self.assertIn(f"[STDERR][FROM_ENV]: {env_input}", stderr_text)

    def stdin_redirection(self, console: Console):
        """Only stdin redirected. Verify every input path via console output."""
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")
        stdin_input = "from stdin"
        args_input = "from argv"
        env_input = "from env"

        with NamedTemporaryFile("w+t") as stdin:
            stdin.write(stdin_input)
            stdin.flush()
            session.launch(
                LaunchArgs(
                    program,
                    stdio=[stdin.name],
                    console=console,
                    args=["--read-stdin", args_input],
                    env={"FROM_ENV": env_input},
                )
            )
            session.verify_process_exited()

            stdout_text = self._get_debuggee_stdout(session)
            stderr_text = self._get_debuggee_stderr(session)
            self.assertIn(f"[STDOUT][FROM_STDIN]: {stdin_input}", stdout_text)
            self.assertIn(f"[STDOUT][FROM_ARGV]: {args_input}", stdout_text)
            self.assertIn(f"[STDOUT][FROM_ENV]: {env_input}", stdout_text)

            self.assertIn(f"[STDERR][FROM_STDIN]: {stdin_input}", stderr_text)
            self.assertIn(f"[STDERR][FROM_ARGV]: {args_input}", stderr_text)
            self.assertIn(f"[STDERR][FROM_ENV]: {env_input}", stderr_text)

    def stdout_redirection(self, console: Console):
        """Only stdout redirected. Verify argv and env paths.

        stdin is not set up — the C++ program skips reading it because the
        file descriptor is a tty (would block).
        """
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")
        args_input = "from argv"
        env_input = "from env"

        with NamedTemporaryFile("rt") as stdout:
            session.launch(
                LaunchArgs(
                    program,
                    stdio=[None, stdout.name],
                    console=console,
                    args=[args_input],
                    env={"FROM_ENV": env_input},
                )
            )
            session.verify_process_exited()

            stdout_text = stdout.read()
            stderr_text = self._get_debuggee_stderr(session)
            self.assertIn(f"[STDOUT][FROM_ARGV]: {args_input}", stdout_text)
            self.assertIn(f"[STDOUT][FROM_ENV]: {env_input}", stdout_text)

            self.assertIn(f"[STDERR][FROM_ARGV]: {args_input}", stderr_text)
            self.assertIn(f"[STDERR][FROM_ENV]: {env_input}", stderr_text)

    def stderr_redirection(self, console: Console):
        """Only stderr redirected. Verify argv and env paths."""
        session = self.build_and_create_session()
        program = self.getBuildArtifact("a.out")
        args_input = "from argv"
        env_input = "from env"

        with NamedTemporaryFile("rt") as stderr:
            session.launch(
                LaunchArgs(
                    program,
                    stdio=[None, None, stderr.name],
                    console=console,
                    args=[args_input],
                    env={"FROM_ENV": env_input},
                )
            )
            session.verify_process_exited()

            stdout_text = self._get_debuggee_stdout(session)
            stderr_text = stderr.read()
            self.assertIn(f"[STDOUT][FROM_ARGV]: {args_input}", stdout_text)
            self.assertIn(f"[STDOUT][FROM_ENV]: {env_input}", stdout_text)

            self.assertIn(f"[STDERR][FROM_ARGV]: {args_input}", stderr_text)
            self.assertIn(f"[STDERR][FROM_ENV]: {env_input}", stderr_text)

    @abstractmethod
    def _get_debuggee_stdout(self, session: DAPTestSession) -> str:
        """Retrieves the standard output (stdout) from the debuggee process.

        The default destination of the debuggee's stdout can vary based on how the debuggee
        was launched (either a debug console or a pseudo-terminal (pty)).
        It requires subclasses to implement the specific mechanism for obtaining the stdout stream.
        """
        raise RuntimeError(f"NotImplemented for {self}")

    @abstractmethod
    def _get_debuggee_stderr(self, session: DAPTestSession) -> str:
        """Retrieves the standard error (stderr) from the debuggee process.

        The default destination of the debuggee's stderr can vary based on how the debuggee
        was launched (either a debug console or a pseudo-terminal (pty)).
        It requires subclasses to implement the specific mechanism for obtaining the stderr stream.
        """
        raise RuntimeError(f"NotImplemented for {self}")
