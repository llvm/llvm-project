import getopt
import os, subprocess
import stat
import pathlib
import platform
import shutil
from io import StringIO

from lit.ShellEnvironment import expand_glob_expressions, InternalShellError, kIsWindows, processRedirects, ShellCommandResult, updateEnv
import lit.util

def executeBuiltinCd(cmd, shenv):
    """executeBuiltinCd - Change the current directory."""
    if len(cmd.args) != 2:
        raise InternalShellError(cmd, "'cd' supports only one argument")
    # Update the cwd in the parent environment.
    shenv.change_dir(cmd.args[1])
    # The cd builtin always succeeds. If the directory does not exist, the
    # following Popen calls will fail instead.
    return ShellCommandResult(cmd, "", "", 0, False)


def executeBuiltinPushd(cmd, shenv):
    """executeBuiltinPushd - Change the current dir and save the old."""
    if len(cmd.args) != 2:
        raise InternalShellError(cmd, "'pushd' supports only one argument")
    shenv.dirStack.append(shenv.cwd)
    shenv.change_dir(cmd.args[1])
    return ShellCommandResult(cmd, "", "", 0, False)


def executeBuiltinPopd(cmd, shenv):
    """executeBuiltinPopd - Restore a previously saved working directory."""
    if len(cmd.args) != 1:
        raise InternalShellError(cmd, "'popd' does not support arguments")
    if not shenv.dirStack:
        raise InternalShellError(cmd, "popd: directory stack empty")
    shenv.cwd = shenv.dirStack.pop()
    return ShellCommandResult(cmd, "", "", 0, False)


def executeBuiltinExport(cmd, shenv):
    """executeBuiltinExport - Set an environment variable."""
    if len(cmd.args) != 2:
        raise InternalShellError(cmd, "'export' supports only one argument")
    updateEnv(shenv, cmd.args)
    return ShellCommandResult(cmd, "", "", 0, False)


def executeBuiltinEcho(cmd, shenv):
    """Interpret a redirected echo or @echo command"""
    opened_files = []
    stdin, stdout, stderr = processRedirects(cmd, subprocess.PIPE, shenv, opened_files)
    if stdin != subprocess.PIPE or stderr != subprocess.PIPE:
        raise InternalShellError(
            cmd, f"stdin and stderr redirects not supported for {cmd.args[0]}"
        )

    # Some tests have un-redirected echo commands to help debug test failures.
    # Buffer our output and return it to the caller.
    is_redirected = True
    if stdout == subprocess.PIPE:
        is_redirected = False
        stdout = StringIO()
    elif kIsWindows:
        # Reopen stdout with `newline=""` to avoid CRLF translation.
        # The versions of echo we are replacing on Windows all emit plain LF,
        # and the LLVM tests now depend on this.
        stdout = open(stdout.name, stdout.mode, encoding="utf-8", newline="")
        opened_files.append((None, None, stdout, None))

    # Implement echo flags. We only support -e and -n, and not yet in
    # combination. We have to ignore unknown flags, because `echo "-D FOO"`
    # prints the dash.
    args = cmd.args[1:]
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

    for (name, mode, f, path) in opened_files:
        f.close()

    output = "" if is_redirected else stdout.getvalue()
    return ShellCommandResult(cmd, output, "", 0, False)


def executeBuiltinMkdir(cmd, cmd_shenv):
    """executeBuiltinMkdir - Create new directories."""
    args = expand_glob_expressions(cmd.args, cmd_shenv.cwd)[1:]
    try:
        opts, args = getopt.gnu_getopt(args, "p")
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

    stderr = StringIO()
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
                stderr.write("Error: 'mkdir' command failed, %s\n" % str(err))
                exitCode = 1
    return ShellCommandResult(cmd, "", stderr.getvalue(), exitCode, False)


def executeBuiltinRm(cmd, cmd_shenv):
    """executeBuiltinRm - Removes (deletes) files or directories."""
    args = expand_glob_expressions(cmd.args, cmd_shenv.cwd)[1:]
    try:
        opts, args = getopt.gnu_getopt(args, "frR", ["--recursive"])
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

    stderr = StringIO()
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
                    stderr.write("Error: %s is a directory\n" % path)
                    exitCode = 1
                if platform.system() == "Windows":
                    # NOTE: use ctypes to access `SHFileOperationsW` on Windows to
                    # use the NT style path to get access to long file paths which
                    # cannot be removed otherwise.
                    from ctypes.wintypes import BOOL, HWND, LPCWSTR, UINT, WORD
                    from ctypes import addressof, byref, c_void_p, create_unicode_buffer
                    from ctypes import Structure
                    from ctypes import windll, WinError, POINTER

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
            stderr.write("Error: 'rm' command failed, %s" % str(err))
            exitCode = 1
    return ShellCommandResult(cmd, "", stderr.getvalue(), exitCode, False)


def executeBuiltinUmask(cmd, shenv):
    """executeBuiltinUmask - Change the current umask."""
    if os.name != "posix":
        raise InternalShellError(cmd, "'umask' not supported on this system")
    if len(cmd.args) != 2:
        raise InternalShellError(cmd, "'umask' supports only one argument")
    try:
        # Update the umask in the parent environment.
        shenv.umask = int(cmd.args[1], 8)
    except ValueError as err:
        raise InternalShellError(cmd, "Error: 'umask': %s" % str(err))
    return ShellCommandResult(cmd, "", "", 0, False)


def executeBuiltinUlimit(cmd, shenv):
    """executeBuiltinUlimit - Change the current limits."""
    try:
        # Try importing the resource module (available on POSIX systems) and
        # emit an error where it does not exist (e.g., Windows).
        import resource
    except ImportError:
        raise InternalShellError(cmd, "'ulimit' not supported on this system")
    if len(cmd.args) != 3:
        raise InternalShellError(cmd, "'ulimit' requires two arguments")
    try:
        if cmd.args[2] == "unlimited":
            new_limit = resource.RLIM_INFINITY
        else:
            new_limit = int(cmd.args[2])
    except ValueError as err:
        raise InternalShellError(cmd, "Error: 'ulimit': %s" % str(err))
    if cmd.args[1] == "-v":
        if new_limit != resource.RLIM_INFINITY:
            new_limit = new_limit * 1024
        shenv.ulimit["RLIMIT_AS"] = new_limit
    elif cmd.args[1] == "-n":
        shenv.ulimit["RLIMIT_NOFILE"] = new_limit
    elif cmd.args[1] == "-s":
        if new_limit != resource.RLIM_INFINITY:
            new_limit = new_limit * 1024
        shenv.ulimit["RLIMIT_STACK"] = new_limit
    elif cmd.args[1] == "-f":
        shenv.ulimit["RLIMIT_FSIZE"] = new_limit
    else:
        raise InternalShellError(
            cmd, "'ulimit' does not support option: %s" % cmd.args[1]
        )
    return ShellCommandResult(cmd, "", "", 0, False)


def executeBuiltinColon(cmd, cmd_shenv):
    """executeBuiltinColon - Discard arguments and exit with status 0."""
    return ShellCommandResult(cmd, "", "", 0, False)
