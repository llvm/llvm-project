from __future__ import annotations

import os
import subprocess
from threading import Thread
from typing import Any, Callable, Optional, Tuple

try:
    from queue import Empty, Queue
except ImportError:
    from Queue import Empty, Queue

from lit.InprocBuiltins import InprocBuiltinIO
from lit.ShCommands import Command
from lit.ShellEnvironment import ShellEnvironment, kIsWindows

if kIsWindows:
    import msvcrt

    # Required for `set_blocking`.
    from ctypes import POINTER, WinError, byref, c_wchar, windll
    from ctypes.wintypes import BOOL, DWORD, HANDLE


debug = os.environ.get("LLVM_LIT_TRACE_DAEMON_COMMUNICATION", "0") == "1"
"""
Output a trace of data sent to and received from the daemon process.
"""


def remove_prefix(string: str, prefix: str) -> str:
    # FIXME: Once Python 3.9+ is required, replace uses of this with
    # `str.removeprefix`.
    if string.startswith(prefix):
        return string[len(prefix) :]
    return string


def set_blocking(pipefd: int, blocking: bool):
    """
    Cross-platform replacement for `os.set_blocking`.
    """
    # `os.set_blocking` is not available on Windows until Python 3.12, so
    # we reimplement it using SetNamedPipeHandleState, as is used in the
    # CPython implementation:
    #   https://github.com/python/cpython/blob/6304eb1f5b93f682bff558befe4a7b9585f4601e/Python/fileutils.c#L2849
    #
    # FIXME: Once Python 3.12+ is required, replace uses of this with
    # `os.setblocking`

    if not kIsWindows:
        os.set_blocking(pipefd, blocking)
        return

    handle = msvcrt.get_osfhandle(pipefd)
    mode = DWORD()

    windll.kernel32.GetNamedPipeHandleStateW.restype = BOOL
    windll.kernel32.GetNamedPipeHandleStateW.argtypes = [
        HANDLE,  # hNamedPipe
        POINTER(DWORD),  # [optional] lpState
        POINTER(DWORD),  # [optional] lpCurInstances
        POINTER(DWORD),  # [optional] lpMaxCollectionCount
        POINTER(DWORD),  # [optional] lpCollectDataTimeout
        POINTER(c_wchar),  # [optional] lpUserName
        DWORD,  # nMaxUserNameSize
    ]
    ok = windll.kernel32.GetNamedPipeHandleStateW(
        handle,
        byref(mode),
        None,
        None,
        None,
        None,
        0,
    )
    if not ok:
        raise WinError()

    PIPE_NOWAIT = 1
    if blocking:
        mode.value &= ~PIPE_NOWAIT
    else:
        mode.value |= PIPE_NOWAIT

    windll.kernel32.SetNamedPipeHandleState.restype = BOOL
    windll.kernel32.SetNamedPipeHandleState.argtypes = [
        HANDLE,  # hNamedPipe
        POINTER(DWORD),  # [optional] lpMode
        POINTER(DWORD),  # [optional] lpMaxCollectionCount
        POINTER(DWORD),  # [optional] lpCollectDataTimeout
    ]
    ok = windll.kernel32.SetNamedPipeHandleState(
        handle,
        byref(mode),
        None,
        None,
    )
    if not ok:
        raise WinError()


class DaemonError(Exception):
    """
    Exception raised when the daemon tool sends an error message.
    """

    def __init__(self, message: str):
        super().__init__()
        self.message = remove_prefix(message, "error ")

    def __str__(self) -> str:
        return f"Error from daemon: {self.message}"


class UnexpectedDaemonOutput(Exception):
    """
    Exception raised when the daemon tool sends an unexpected message.
    """

    def __init__(self, message: bytes):
        super().__init__()
        self.message = message

    def __str__(self) -> str:
        return f"Unexpected message from daemon: {self.message}"


class DaemonExited(Exception):
    """
    Exception raised when the daemon exits unexpectedly.
    """

    exit_code: int
    stdout: bytes
    stderr: bytes

    def __init__(self, exit_code: int, stdout: bytes, stderr: bytes):
        super().__init__()
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self) -> str:
        return "Daemon exited unexpectedly with code {}.\nstdout:\n{}\nstderr:\n{}\n".format(
            self.exit_code, self.stdout, self.stderr
        )


def async_read_status_pipe(daemon):
    """
    Runs in the background, reading from the status pipe from the daemon
    process and writing it to a queue. We use a queue so that the pipe can be
    read with blocking (via `get`) and without (via `get_nowait`)
    """

    while daemon.is_alive():
        line = daemon.status_pipe.readline()
        if debug:
            print("from status pipe:", line)
        daemon.status_pipe_queue.put(line)
    daemon.status_pipe_queue.put(b"")


class DaemonTool:
    executable_path: str
    daemon_proc: Optional[subprocess.Popen]
    status_pipe: Any
    status_pipe_queue: Queue
    status_pipe_reader_thread: Optional[Thread]

    def __init__(self, executable_path: str):
        self.executable_path = executable_path
        self.daemon_proc = None
        self.status_pipe = None
        self.status_pipe_queue = Queue()
        self.status_pipe_reader_thread = None

    def is_alive(self):
        if not self.daemon_proc:
            return False
        if not isinstance(self.daemon_proc, subprocess.Popen):
            return False

        return self.daemon_proc.poll() is None

    def start_daemon(self):
        assert not self.is_alive(), "start_daemon called but daemon is already alive."

        # Close the old status pipe.
        if self.status_pipe:
            self.status_pipe.close()

        # Kill the old status pipe reading thread.
        if self.status_pipe_reader_thread:
            self.status_pipe_reader_thread.join()

        # Clear the status pipe queue, to avoid issues caused by lingering
        # messages.
        self.status_pipe_queue = Queue()

        # Create a new status pipe for the daemon process.
        # This will be used by the daemon to communicate its status, including
        # exit codes.
        status_pipe_reader, status_pipe_writer = os.pipe()

        # Make sure that the write end of the status pipe gets inherited
        # by the daemon.
        os.set_inheritable(status_pipe_writer, True)
        if kIsWindows:
            status_pipe_handle = msvcrt.get_osfhandle(status_pipe_writer)
            os.set_handle_inheritable(status_pipe_handle, True)

        args = [
            self.executable_path,
            "--daemon",
        ]
        # On Windows, only the file handle (not the file descriptor) is
        # inherited.
        if kIsWindows:
            args.append(f"--daemon-status-pipe=handle:{status_pipe_handle}")
            startupinfo = subprocess.STARTUPINFO(
                lpAttributeList={"handle_list": [status_pipe_handle]},
            )
            pass_fds = ()
        else:
            args.append(f"--daemon-status-pipe=fd:{status_pipe_writer}")
            startupinfo = None
            pass_fds = [status_pipe_writer]

        self.daemon_proc = subprocess.Popen(
            args=args,
            text=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=pass_fds,
            startupinfo=startupinfo,
        )

        # Close our status pipe writer, as we only read from the pipe.
        os.close(status_pipe_writer)

        set_blocking(self.daemon_proc.stdout.fileno(), False)
        set_blocking(self.daemon_proc.stderr.fileno(), False)

        # Start the status pipe reader thread.
        self.status_pipe = open(status_pipe_reader, "rb")
        self.status_pipe_reader_thread = Thread(
            target=async_read_status_pipe,
            args=[self],
            daemon=True,
        )
        self.status_pipe_reader_thread.start()

        # Check initialization status.
        self.expect_response(b"ready")

    def invoke(
        self,
        args: list[str],
        shenv: ShellEnvironment,
        stdin: Any,
        stdout: Any,
        stderr: Any,
    ) -> int:
        # Ensure the daemon is alive.
        if not self.is_alive():
            self.start_daemon()

        commands: list[str | bytes] = []

        # Set the CWD for the daemon.
        abs_cwd = os.path.abspath(shenv.cwd)
        if abs_cwd != os.getcwd():
            commands.append(f"cd {abs_cwd}")

        # Set the arguments for the tool.
        for arg in args:
            arg_bytes = arg.encode()
            commands.append(f"arg {len(arg_bytes)}")
            commands.append(arg_bytes)

        # Set the input for the tool.
        if stdin.get_filename():
            # If the input is a named file, we provide it to the daemon via the
            # "input_file" command.
            commands.append(f"input_file {stdin.get_filename()}")
        else:
            # Otherwise, read the input and provide it via stdin.
            stdin_bytes = stdin.read_binary()
            if stdin_bytes:
                commands.append(f"input_string {len(stdin_bytes)}")
                commands.append(stdin_bytes)
        # If stdout and stderr are the same stream (which will be the case if
        # stderr is redirected to stdout), inform the daemon to send stderr
        # over stdout. We do this to make sure that the order of output is
        # preserved.
        if stderr == stdout:
            commands.append("redirect_stderr_to_stdout")

        commands.append("run")
        self.send_commands(commands)
        exit_code, stdout_bytes, stderr_bytes = self.collect_invocation_output()

        # Make sure the streams are flushed.
        if stderr == stdout:
            stdout.write_binary(stdout_bytes)
            stdout.flush()
        else:
            stdout.write_binary(stdout_bytes)
            stdout.flush()
            stderr.write_binary(stderr_bytes)
            stderr.flush()

        return exit_code

    def collect_invocation_output(self) -> Tuple[int, bytes, bytes]:
        # Wait for a message on the status pipe, indicating the result of the
        # command, while continually reading all output from the daemon on
        # its output streams. It is important to read the output continually
        # rather than reading it all at the end to avoid deadlock if the pipe
        # becomes full.
        message = b""
        stdout = b""
        stderr = b""
        while self.is_alive():
            # Read output from stdout and stderr so far.
            stdout += self.read_output_so_far(self.daemon_proc.stdout)
            stderr += self.read_output_so_far(self.daemon_proc.stderr)

            # Check for a message from the status pipe, indicating that the
            # task has finished.
            try:
                # It is important not to block forever, as this would cause
                # the worker to hang if the pipe becomes full. However, it is
                # also important to wait a bit, as otherwise Lit spinning in
                # this loop consumes significant CPU time, reducing the testing
                # performance. The timeout of 0.01 seconds seems to be a good
                # compromise.
                message = self.status_pipe_queue.get(block=True, timeout=0.01)
                break
            except Empty:
                continue

        # Make sure to read the remainder of the bytes in the output streams.
        stdout += self.read_output_so_far(self.daemon_proc.stdout)
        stderr += self.read_output_so_far(self.daemon_proc.stderr)

        # Check that the message indicates that the task was completed.
        try:
            self.check_response(
                message,
                lambda message: message.startswith(b"returned"),
            )

            # The exit code is stored in the message.
            exit_code = int(remove_prefix(message.decode(), "returned").strip())
            return (exit_code, stdout, stderr)
        except DaemonExited as e:
            # The daemon exited during execution of the task, indicating that
            # the LLVM code crashed or otherwise called `exit`.
            # However the LLVM code exited, the exit code returned by the daemon
            # process is the same as the code that the tool would have returned
            # if run separately, so this is correct to use as the exit code for
            # the test.
            stdout += e.stdout
            stderr += e.stderr
            return (e.exit_code, stdout, stderr)

    def read_output_so_far(self, pipe: Any) -> bytes:
        """
        Read all of the bytes currently in the pipe, which must be in non-
        blocking mode.
        """

        output = b""
        while self.is_alive():
            chunk = pipe.read()
            if not chunk:
                break
            output += chunk

        return output

    def send_commands(self, commands: list[str | bytes]):
        if debug:
            print(f"sending commands: {commands}")
        daemon_input = (
            b"\n".join(
                command.encode() if isinstance(command, str) else command
                for command in commands
            )
            + b"\n"
        )
        os.write(self.daemon_proc.stdin.fileno(), daemon_input)

    def check_response(
        self,
        message: Optional[bytes],
        predicate: Callable[[bytes], bool],
    ):
        """
        Given a message read from the daemon's status pipe, checks that the
        message matches the predicate. Otherwise, raises the appropriate
        exception (DaemonError, DaemonExited, UnexpectedDaemonOutput)
        """

        if not message:
            # On Windows, we must change these streams back to blocking mode
            # for the output to be captured by `communicate()`.
            set_blocking(self.daemon_proc.stdout.fileno(), True)
            set_blocking(self.daemon_proc.stderr.fileno(), True)

            stdout, stderr = self.daemon_proc.communicate()
            raise DaemonExited(self.daemon_proc.returncode, stdout, stderr)

        if predicate(message.strip()):
            return

        if message.startswith(b"error"):
            raise DaemonError(message.decode())

        raise UnexpectedDaemonOutput(message)

    def expect_response(self, expected: bytes):
        self.check_response(
            self.status_pipe_queue.get(), lambda received: received == expected
        )

    def close(self):
        if not self.is_alive():
            return

        self.send_commands(["exit"])

        try:
            self.daemon_proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.daemon_proc.kill()


daemons: dict[str, DaemonTool] = {}
"""
Dictionary storing the existing daemon for a given tool path. This is how
existing daemons are remembered between invocations.
"""


def invoke_llvm_daemon_tool(
    executable_path: str,
    cmd: Command,
    args: list[str],
    shenv: ShellEnvironment,
    io: InprocBuiltinIO,
):
    """
    Function called by the in-process builtins that are invoking daemon tools.
    """

    # Find the daemon corresponding to this tool executable, or create one.
    daemon = daemons.get(executable_path, None)
    if not daemon:
        daemon = DaemonTool(executable_path)
        daemons[executable_path] = daemon

    args[0] = executable_path
    return daemon.invoke(args, shenv, io.stdin, io.stdout, io.stderr)
