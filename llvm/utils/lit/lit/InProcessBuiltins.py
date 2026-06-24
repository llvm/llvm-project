from __future__ import annotations

import abc
import getopt
import io
import os
import pathlib
import shutil
import stat
import subprocess
from dataclasses import dataclass
from typing import Callable

import lit.util
from lit.ShCommands import Command
from lit.ShellEnvironment import (
    InternalShellError,
    ShellEnvironment,
    kIsWindows,
    updateEnv,
)


class InProcessBuiltinIOObject(abc.ABC):
    """
    Base class for IO streams used for in-process built-ins. This class has two
    specializations: InProcessBuiltinIOFile, wrapping a file open in text mode,
    and InProcessBuiltinIOMemory, wrapping a BytesIO object. These streams
    provide a text IO interface, but support reading and writing binary data
    too.

    The main reason that this exists is to solve the conflict that binary IO is
    required for daemonized testing, as many tests involve tools reading and
    writing binary files and piping binary data between themselves, but on z/OS
    files must be opened in text mode so that the character encoding is correct.
    So we need an IO object that can be both a file open in text mode and a
    in-memory stream that can store binary data.
    """

    @abc.abstractmethod
    def read(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def read_binary(self) -> bytes:
        raise NotImplementedError()

    @abc.abstractmethod
    def write(self, data: str) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def write_binary(self, data: bytes) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def seek(self, pos: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def flush(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_filename(self) -> str | None:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_encoding(self) -> str | None:
        raise NotImplementedError()


@dataclass
class InProcessBuiltinIOFile(InProcessBuiltinIOObject):
    """
    Specialization of InProcessBuiltinIOObject wrapping a file which is open in
    text mode. Files must be opened in text mode so that character encoding
    tags are correctly handled on z/OS. Binary IO is still supported via the
    OS APIs (needed for daemonized testing).
    """

    file: io.TextIOBase

    def read(self) -> str:
        return self.file.read()

    def read_binary(self) -> bytes:
        self.file.flush()
        data = bytearray()
        chunk_size = 1024

        while True:
            chunk = os.read(self.file.fileno(), chunk_size)
            if not chunk:
                break
            data.extend(chunk)

        return bytes(data)

    def write(self, data: str) -> int:
        return self.file.write(data)

    def write_binary(self, data: bytes) -> int:
        self.file.flush()
        return os.write(self.file.fileno(), data)

    def seek(self, pos: int):
        self.file.seek(pos)

    def flush(self):
        self.file.flush()

    def get_filename(self) -> str | None:
        if hasattr(self.file, "name") and isinstance(self.file.name, str):
            return self.file.name
        return None

    def get_encoding(self) -> str | None:
        return str(self.file.encoding)


@dataclass
class InProcessBuiltinIOMemory(InProcessBuiltinIOObject):
    """
    Specialization of InProcessBuiltinIOObject wrapping an in-memory binary stream.
    """

    obj: io.BytesIO
    encoding: str = "utf-8"

    def read(self) -> str:
        return self.obj.read().decode(self.encoding, errors="replace")

    def read_binary(self) -> bytes:
        return self.obj.read()

    def write(self, data: str) -> int:
        return self.obj.write(data.encode(self.encoding, errors="replace"))

    def write_binary(self, data: bytes) -> int:
        return self.obj.write(data)

    def seek(self, pos: int):
        self.obj.seek(pos)

    def flush(self):
        pass

    def get_filename(self) -> str | None:
        return None

    def get_encoding(self) -> str | None:
        return self.encoding


class InProcessBuiltinIOStreams:
    """
    Holds IO streams for an inproc builtin invocation.

    NB: If stderr is redirected to be the same stream as stdout, then
    `stder == stdout` is True.
    """

    stdin: InProcessBuiltinIOObject
    stdout: InProcessBuiltinIOObject
    stderr: InProcessBuiltinIOObject

    def __init__(self, stdin, stdout, stderr):
        """
        Configure the IO streams for an in-process builtin command in
        the same way that IO streams are configured when calling
        `subprocess.Popen`.

        Each of stdin, stdout and stderr may be:
        - A file object open in binary mode.
        - `subprocess.PIPE`
        - `subprocess.STDOUT` (for stderr)
        - None
        """

        # If stderr is redirected to stdout, we make sure to use the same
        # stream for both so that the order of output is preserved.
        stderr_redirected_to_stdout = (
            stdout == subprocess.PIPE and stderr == subprocess.STDOUT
        )

        # Replace sentinel values with in-memory streams.
        def resolve_io_obj(stream) -> InProcessBuiltinIOObject:
            if stream == subprocess.PIPE or stream is None:
                return InProcessBuiltinIOMemory(io.BytesIO())
            elif isinstance(stream, InProcessBuiltinIOObject):
                return stream
            else:
                return InProcessBuiltinIOFile(stream)

        self.stdin = resolve_io_obj(stdin)
        self.stdout = resolve_io_obj(stdout)
        if stderr_redirected_to_stdout:
            # Make sure stderr and stdout are directed to the same stream.
            self.stderr = self.stdout
        else:
            self.stderr = resolve_io_obj(stderr)


# Function called by an in-process builtin command. The return value is the
# exit code.
# Parameters:
# - `cmd`: The command itself.
# - `args`: glob-expanded list of arguments (including argv[0] as the program name).
# - `shenv`: The shell environment.
# - `io`: Holds the input and output streams for the invocation. These are file-like objects (files, StringIO)
InProcessBuiltinExecuteFn = Callable[
    [Command, "list[str]", ShellEnvironment, InProcessBuiltinIOStreams],
    int,
]


@dataclass
class InProcessBuiltin:
    """
    Represents a command that is run as an in-process builtins.
    """

    execute: InProcessBuiltinExecuteFn


def executeBuiltinCd(
    cmd: Command,
    args: list[str],
    shenv: ShellEnvironment,
    io: InProcessBuiltinIOStreams,
) -> int:
    """executeBuiltinCd - Change the current directory."""
    if len(args) != 2:
        raise InternalShellError(cmd, "'cd' supports only one argument")
    # Update the cwd in the parent environment.
    shenv.change_dir(args[1])
    # The cd builtin always succeeds. If the directory does not exist, the
    # following Popen calls will fail instead.
    return 0


def executeBuiltinPushd(
    cmd: Command,
    args: list[str],
    shenv: ShellEnvironment,
    io: InProcessBuiltinIOStreams,
) -> int:
    """executeBuiltinPushd - Change the current dir and save the old."""
    if len(args) != 2:
        raise InternalShellError(cmd, "'pushd' supports only one argument")
    shenv.dirStack.append(shenv.cwd)
    shenv.change_dir(args[1])
    return 0


def executeBuiltinPopd(
    cmd: Command,
    args: list[str],
    shenv: ShellEnvironment,
    io: InProcessBuiltinIOStreams,
) -> int:
    """executeBuiltinPopd - Restore a previously saved working directory."""
    if len(args) != 1:
        raise InternalShellError(cmd, "'popd' does not support arguments")
    if not shenv.dirStack:
        raise InternalShellError(cmd, "popd: directory stack empty")
    shenv.cwd = shenv.dirStack.pop()
    return 0


def executeBuiltinExport(
    cmd: Command,
    args: list[str],
    shenv: ShellEnvironment,
    io: InProcessBuiltinIOStreams,
) -> int:
    """executeBuiltinExport - Set an environment variable."""
    if len(args) != 2:
        raise InternalShellError(cmd, "'export' supports only one argument")
    updateEnv(shenv, args)
    return 0


def executeBuiltinEcho(
    cmd: Command,
    args: list[str],
    shenv: ShellEnvironment,
    io: InProcessBuiltinIOStreams,
) -> int:
    """Interpret a redirected echo or @echo command"""
    opened_files = []

    stdout = io.stdout
    if (
        kIsWindows
        and isinstance(io.stdout, InProcessBuiltinIOFile)
        and io.stdout.get_filename()
    ):
        # Reopen stdout with `newline=""` to avoid CRLF translation.
        # The versions of echo we are replacing on Windows all emit plain LF,
        # and the LLVM tests now depend on this.
        stdout = open(
            io.stdout.get_filename(),
            io.stdout.file.mode,
            encoding="utf-8",
            newline="",
        )
        opened_files.append((None, None, stdout, None))

    # Implement echo flags. We only support -e and -n, and not yet in
    # combination. We have to ignore unknown flags, because `echo "-D FOO"`
    # prints the dash.
    args = args[1:]
    interpret_escapes = False
    write_newline = True
    while len(args) >= 1 and args[0] in ("-e", "-n"):
        flag = args[0]
        args = args[1:]
        if flag == "-e":
            interpret_escapes = True
        elif flag == "-n":
            write_newline = False

    def maybeUnescape(arg):
        if not interpret_escapes:
            return arg

        return arg.encode("utf-8").decode("unicode_escape")

    if args:
        for arg in args[:-1]:
            stdout.write(maybeUnescape(arg))
            stdout.write(" ")
        stdout.write(maybeUnescape(args[-1]))
    if write_newline:
        stdout.write("\n")

    for name, mode, f, path in opened_files:
        f.close()

    return 0


def executeBuiltinMkdir(
    cmd: Command,
    args: list[str],
    cmd_shenv: ShellEnvironment,
    io: InProcessBuiltinIOStreams,
):
    """executeBuiltinMkdir - Create new directories."""
    try:
        opts, args = getopt.gnu_getopt(args[1:], "p")
    except getopt.GetoptError as err:
        raise InternalShellError(cmd, "Unsupported: 'mkdir':  %s" % str(err))

    parent = False
    for o, a in opts:
        if o == "-p":
            parent = True
        else:
            assert False, "unhandled option"

    if len(args) == 0:
        raise InternalShellError(cmd, "Error: 'mkdir' is missing an operand")

    exitCode = 0
    for dir in args:
        dir = pathlib.Path(dir)
        cwd = pathlib.Path(cmd_shenv.cwd)
        if not dir.is_absolute():
            dir = lit.util.abs_path_preserve_drive(cwd / dir)
        if parent:
            dir.mkdir(parents=True, exist_ok=True)
        else:
            try:
                dir.mkdir(exist_ok=True)
            except OSError as err:
                io.stderr.write("Error: 'mkdir' command failed, %s\n" % str(err))
                exitCode = 1
    return exitCode


def executeBuiltinRm(
    cmd: Command,
    args: list[str],
    cmd_shenv: ShellEnvironment,
    io: InProcessBuiltinIOStreams,
):
    """executeBuiltinRm - Removes (deletes) files or directories."""
    try:
        opts, args = getopt.gnu_getopt(args[1:], "frR", ["--recursive"])
    except getopt.GetoptError as err:
        raise InternalShellError(cmd, "Unsupported: 'rm':  %s" % str(err))

    force = False
    recursive = False
    for o, a in opts:
        if o == "-f":
            force = True
        elif o in ("-r", "-R", "--recursive"):
            recursive = True
        else:
            assert False, "unhandled option"

    if len(args) == 0:
        raise InternalShellError(cmd, "Error: 'rm' is missing an operand")

    def on_rm_error(func, path, exc_info):
        # path contains the path of the file that couldn't be removed
        # let's just assume that it's read-only and remove it.
        os.chmod(path, stat.S_IMODE(os.stat(path).st_mode) | stat.S_IWRITE)
        os.remove(path)

    exitCode = 0
    for path in args:
        cwd = cmd_shenv.cwd
        if not os.path.isabs(path):
            path = lit.util.abs_path_preserve_drive(os.path.join(cwd, path))
        if force and not os.path.exists(path):
            continue
        try:
            if os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                if not recursive:
                    io.stderr.write("Error: %s is a directory\n" % path)
                    exitCode = 1
                if kIsWindows:
                    # NOTE: use ctypes to access `SHFileOperationsW` on Windows to
                    # use the NT style path to get access to long file paths which
                    # cannot be removed otherwise.
                    from ctypes import (
                        POINTER,
                        Structure,
                        WinError,
                        addressof,
                        byref,
                        c_void_p,
                        create_unicode_buffer,
                        windll,
                    )
                    from ctypes.wintypes import BOOL, HWND, LPCWSTR, UINT, WORD

                    class SHFILEOPSTRUCTW(Structure):
                        _fields_ = [
                            ("hWnd", HWND),
                            ("wFunc", UINT),
                            ("pFrom", LPCWSTR),
                            ("pTo", LPCWSTR),
                            ("fFlags", WORD),
                            ("fAnyOperationsAborted", BOOL),
                            ("hNameMappings", c_void_p),
                            ("lpszProgressTitle", LPCWSTR),
                        ]

                    FO_MOVE, FO_COPY, FO_DELETE, FO_RENAME = range(1, 5)

                    FOF_SILENT = 4
                    FOF_NOCONFIRMATION = 16
                    FOF_NOCONFIRMMKDIR = 512
                    FOF_NOERRORUI = 1024

                    FOF_NO_UI = (
                        FOF_SILENT
                        | FOF_NOCONFIRMATION
                        | FOF_NOERRORUI
                        | FOF_NOCONFIRMMKDIR
                    )

                    SHFileOperationW = windll.shell32.SHFileOperationW
                    SHFileOperationW.argtypes = [POINTER(SHFILEOPSTRUCTW)]

                    path = os.path.abspath(path)

                    pFrom = create_unicode_buffer(path, len(path) + 2)
                    pFrom[len(path)] = pFrom[len(path) + 1] = "\0"
                    operation = SHFILEOPSTRUCTW(
                        wFunc=UINT(FO_DELETE),
                        pFrom=LPCWSTR(addressof(pFrom)),
                        fFlags=FOF_NO_UI,
                    )
                    result = SHFileOperationW(byref(operation))
                    if result:
                        raise WinError(result)
                else:
                    shutil.rmtree(path, onerror=on_rm_error if force else None)
            else:
                if force and not os.access(path, os.W_OK):
                    os.chmod(path, stat.S_IMODE(os.stat(path).st_mode) | stat.S_IWRITE)
                os.remove(path)
        except OSError as err:
            io.stderr.write("Error: 'rm' command failed, %s" % str(err))
            exitCode = 1
    return exitCode


def executeBuiltinUmask(
    cmd: Command,
    args: list[str],
    shenv: ShellEnvironment,
    io: InProcessBuiltinIOStreams,
):
    """executeBuiltinUmask - Change the current umask."""
    if os.name != "posix":
        raise InternalShellError(cmd, "'umask' not supported on this system")
    if len(args) != 2:
        raise InternalShellError(cmd, "'umask' supports only one argument")
    try:
        # Update the umask in the parent environment.
        shenv.umask = int(args[1], 8)
    except ValueError as err:
        raise InternalShellError(cmd, "Error: 'umask': %s" % str(err))
    return 0


def executeBuiltinUlimit(
    cmd: Command, args: list[str], shenv, io: InProcessBuiltinIOStreams
):
    """executeBuiltinUlimit - Change the current limits."""
    try:
        # Try importing the resource module (available on POSIX systems) and
        # emit an error where it does not exist (e.g., Windows).
        import resource
    except ImportError:
        raise InternalShellError(cmd, "'ulimit' not supported on this system")
    if len(args) != 3:
        raise InternalShellError(cmd, "'ulimit' requires two arguments")
    try:
        if args[2] == "unlimited":
            new_limit = resource.RLIM_INFINITY
        else:
            new_limit = int(args[2])
    except ValueError as err:
        raise InternalShellError(cmd, "Error: 'ulimit': %s" % str(err))
    if args[1] == "-v":
        if new_limit != resource.RLIM_INFINITY:
            new_limit = new_limit * 1024
        shenv.ulimit["RLIMIT_AS"] = new_limit
    elif args[1] == "-n":
        shenv.ulimit["RLIMIT_NOFILE"] = new_limit
    elif args[1] == "-s":
        if new_limit != resource.RLIM_INFINITY:
            new_limit = new_limit * 1024
        shenv.ulimit["RLIMIT_STACK"] = new_limit
    elif args[1] == "-f":
        shenv.ulimit["RLIMIT_FSIZE"] = new_limit
    else:
        raise InternalShellError(cmd, "'ulimit' does not support option: %s" % args[1])
    return 0


def executeBuiltinColon(
    cmd: Command,
    args: list[str],
    cmd_shenv: ShellEnvironment,
    io: InProcessBuiltinIOStreams,
):
    """executeBuiltinColon - Discard arguments and exit with status 0."""
    return 0


def get_default_inproc_builtins() -> dict[str, InProcessBuiltin]:
    """
    Returns the map of command names to Lit's in-process built-in
    implementations.
    """

    return {
        "@echo": InProcessBuiltin(executeBuiltinEcho),
        "cd": InProcessBuiltin(executeBuiltinCd),
        "export": InProcessBuiltin(executeBuiltinExport),
        "echo": InProcessBuiltin(executeBuiltinEcho),
        "mkdir": InProcessBuiltin(executeBuiltinMkdir),
        "popd": InProcessBuiltin(executeBuiltinPopd),
        "pushd": InProcessBuiltin(executeBuiltinPushd),
        "rm": InProcessBuiltin(executeBuiltinRm),
        "ulimit": InProcessBuiltin(executeBuiltinUlimit),
        "umask": InProcessBuiltin(executeBuiltinUmask),
        ":": InProcessBuiltin(executeBuiltinColon),
    }
