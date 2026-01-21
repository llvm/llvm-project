"""
Test the redirection of stdio.
There are three ways to launch the debuggee
internalConsole, integratedTerminal and externalTerminal.

For the three configurations, we test if we can read data
from environments, stdin and cli arguments.

NOTE: The testcases do not include all possible configurations of
consoles, environments, stdin and cli arguments.
"""

from abc import abstractmethod
from typing import IO
import lldbdap_testcase
from tempfile import NamedTemporaryFile

from lldbsuite.test.decorators import (
    skip,
    skipIfAsan,
    skipIfBuildType,
    skipIfRemote,
    skipIfWindows,
)


class DAP_launchIO(lldbdap_testcase.DAPTestCaseBase):
    """The class holds the implementation different ways to redirect the debuggee I/O streams
    which is configurable from the Derived classes.

    Depending on the console type the output will be in different places.
    It also provides two abstract functions `_get_debuggee_stdout` and `_get_debuggee_stderr`
    that provides the debuggee stdout and stderr.
    """

    def all_redirection(self, console: str, with_args: bool = False):
        """Test all standard io redirection."""
        self.build_and_create_debug_adapter()
        program = self.getBuildArtifact("a.out")
        input_text = "from stdin with redirection"
        args_text = "string from argv"
        program_args = [args_text] if with_args else None

        with NamedTemporaryFile("wt") as stdin, NamedTemporaryFile(
            "rt"
        ) as stdout, NamedTemporaryFile("rt") as stderr:
            stdin.write(input_text)
            stdin.flush()
            self.launch(
                program,
                stdio=[stdin.name, stdout.name, stderr.name],
                console=console,
                args=program_args,
            )
            self.continue_to_exit()

            all_stdout = stdout.read()
            all_stderr = stderr.read()

            if with_args:
                self.assertEqual(f"[STDOUT][FROM_ARGV]: {args_text}", all_stdout)
                self.assertEqual(f"[STDERR][FROM_ARGV]: {args_text}", all_stderr)

                self.assertNotIn(f"[STDOUT][FROM_ARGV]: {args_text}", all_stderr)
                self.assertNotIn(f"[STDERR][FROM_ARGV]: {args_text}", all_stdout)

            else:
                self.assertEqual(f"[STDOUT][FROM_STDIN]: {input_text}", all_stdout)
                self.assertEqual(f"[STDERR][FROM_STDIN]: {input_text}", all_stderr)

                self.assertNotIn(f"[STDERR][FROM_STDIN]: {input_text}", all_stdout)
                self.assertNotIn(f"[STDOUT][FROM_STDIN]: {input_text}", all_stderr)

    def stdin_redirection(self, console: str, with_args: bool = False):
        """Test only stdin redirection."""
        self.build_and_create_debug_adapter()
        program = self.getBuildArtifact("a.out")
        input_text = "string from stdin"
        args_text = "string from argv"
        program_args = [args_text] if with_args else None

        with NamedTemporaryFile("w+t") as stdin:
            stdin.write(input_text)
            stdin.flush()
            self.launch(program, stdio=[stdin.name], console=console, args=program_args)
            self.continue_to_exit()

            stdout_text = self._get_debuggee_stdout()
            stderr_text = self._get_debuggee_stderr()

            if with_args:
                self.assertIn(f"[STDOUT][FROM_ARGV]: {args_text}", stdout_text)
                self.assertIn(f"[STDERR][FROM_ARGV]: {args_text}", stderr_text)
            else:
                self.assertIn(f"[STDOUT][FROM_STDIN]: {input_text}", stdout_text)
                self.assertIn(f"[STDERR][FROM_STDIN]: {input_text}", stderr_text)

    def stdout_redirection(self, console: str, with_env: bool = False):
        """Test only stdout redirection."""
        self.build_and_create_debug_adapter()
        program = self.getBuildArtifact("a.out")

        argv_text = "output with\n multiline"
        # By default unix terminals the ONLCR flag is enabled. which replaces '\n' with '\r\n'
        # see https://man7.org/linux/man-pages/man3/termios.3.html.
        # This does not affect writing to normal files.
        argv_replaced_text = argv_text.replace("\n", "\r\n")

        program_args = [argv_text]
        env_text = "string from env"
        env = {"FROM_ENV": env_text} if with_env else {}

        with NamedTemporaryFile("rt") as stdout:
            self.launch(
                program,
                stdio=[None, stdout.name],
                console=console,
                args=program_args,
                env=env,
            )
            self.continue_to_exit()

            # check stdout
            stdout_text = stdout.read()
            stderr_text = self._get_debuggee_stderr()
            if with_env:
                self.assertIn(f"[STDOUT][FROM_ENV]: {env_text}", stdout_text)
                self.assertIn(f"[STDERR][FROM_ENV]: {env_text}", stderr_text)

                self.assertNotIn(f"[STDERR][FROM_ENV]: {env_text}", stdout_text)
                self.assertNotIn(f"[STDOUT][FROM_ENV]: {env_text}", stderr_text)
            else:
                self.assertIn(f"[STDOUT][FROM_ARGV]: {argv_text}", stdout_text)

                self.assertNotIn(
                    f"[STDERR][FROM_ARGV]: {argv_replaced_text}", stdout_text
                )
                self.assertNotIn(f"[STDOUT][FROM_ARGV]: {argv_text}", stderr_text)

            # check stderr
            stderr_text = self._get_debuggee_stderr()
            # FIXME: when using 'integrated' or 'external' terminal we do not correctly
            # escape newlines that are sent to the terminal.
            if console == "integratedConsole":
                if with_env:
                    self.assertNotIn(f"[STDOUT][FROM_ENV]: {env_text}", stderr_text)
                    self.assertIn(f"[STDERR][FROM_ENV]: {env_text}", stderr_text)
                else:
                    self.assertNotIn(
                        f"[STDOUT][FROM_ARGV]: {argv_replaced_text}", stderr_text
                    )
                    self.assertIn(
                        f"[STDERR][FROM_ARGV]: {argv_replaced_text}", stderr_text
                    )

    def stderr_redirection(self, console: str, with_env: bool = False):
        """Test only stdout redirection."""
        self.build_and_create_debug_adapter()
        program = self.getBuildArtifact("a.out")

        argv_text = "output with\n multiline"
        # By default unix terminals the ONLCR flag is enabled. which replaces '\n' with '\r\n'
        # see https://man7.org/linux/man-pages/man3/termios.3.html.
        # This does not affect writing to normal files.
        # Currently out test implementation for external and integrated Terminal does not run the
        # program through a shell terminal.
        argv_replaced_text = argv_text
        if console == "internalConsole":
            argv_replaced_text = argv_text.replace("\n", "\r\n")
        program_args = [argv_text]
        env_text = "string from env"
        env = {"FROM_ENV": env_text} if with_env else {}

        with NamedTemporaryFile("rt") as stderr:
            self.launch(
                program,
                stdio=[None, None, stderr.name],
                console=console,
                args=program_args,
                env=env,
            )
            self.continue_to_exit()
            stdout_text = self._get_debuggee_stdout()
            stderr_text = stderr.read()
            if with_env:
                self.assertIn(f"[STDOUT][FROM_ENV]: {env_text}", stdout_text)
                self.assertIn(f"[STDERR][FROM_ENV]: {env_text}", stderr_text)
            else:
                self.assertIn(f"[STDOUT][FROM_ARGV]: {argv_replaced_text}", stdout_text)
                self.assertIn(f"[STDERR][FROM_ARGV]: {argv_text}", stderr_text)

    @abstractmethod
    def _get_debuggee_stdout(self) -> str:
        """Retrieves the standard output (stdout) from the debuggee process.

        The default destination of the debuggee's stdout can vary based on how the debugger
        was launched (either a debug console or a pseudo-terminal (pty)).
        It requires subclasses to implement the specific mechanism for obtaining the stdout stream.
        """
        raise RuntimeError(f"NotImplemented for {self}")

    @abstractmethod
    def _get_debuggee_stderr(self) -> str:
        """Retrieves the standard error (stderr) from the debuggee process.

        The default destination of the debuggee's stderr can vary based on how the debugger
        was launched (either a debug console or a pseudo-terminal (pty)).
        It requires subclasses to implement the specific mechanism for obtaining the stderr stream.
        """
        raise RuntimeError(f"NotImplemented for {self}")


@skipIfWindows
class TestDAP_launchInternalConsole(DAP_launchIO):
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


@skipIfRemote
@skipIfAsan
@skipIfBuildType(["debug"])
@skipIfWindows
class TestDAP_launchIntegratedTerminal(DAP_launchIO):
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


@skip  # NOTE: Currently there is no difference between internal and externalTerminal.
@skipIfRemote
@skipIfAsan
@skipIfBuildType(["debug"])
@skipIfWindows
class TestDAP_launchExternalTerminal(DAP_launchIO):
    console = "externalTerminal"

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
