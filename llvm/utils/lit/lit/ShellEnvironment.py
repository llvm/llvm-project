import os
import platform
import subprocess
import tempfile

import lit.util
from lit.ShCommands import GlobItem

kIsWindows = platform.system() == "Windows"

# Don't use close_fds on Windows.
kUseCloseFDs = not kIsWindows

# Use temporary files to replace /dev/null on Windows.
kAvoidDevNull = kIsWindows
kDevNull = "/dev/null"


class ShellCommandResult(object):
    """Captures the result of an individual command."""

    def __init__(
        self, command, stdout, stderr, exitCode, timeoutReached, outputFiles=[]
    ):
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.exitCode = exitCode
        self.timeoutReached = timeoutReached
        self.outputFiles = list(outputFiles)


class InternalShellError(Exception):
    def __init__(self, command, message):
        self.command = command
        self.message = message


class ShellEnvironment(object):
    """Mutable shell environment containing things like CWD and env vars.

    Environment variables are not implemented, but cwd tracking is. In addition,
    we maintain a dir stack for pushd/popd.
    """

    def __init__(self, cwd, env, umask=-1, ulimit=None):
        self.cwd = cwd
        self.env = dict(env)
        self.umask = umask
        self.dirStack = []
        self.ulimit = ulimit if ulimit else {}

    def change_dir(self, newdir):
        if os.path.isabs(newdir):
            self.cwd = newdir
        else:
            self.cwd = lit.util.abs_path_preserve_drive(os.path.join(self.cwd, newdir))


# args are from 'export' or 'env' command.
# Skips the command, and parses its arguments.
# Modifies env accordingly.
# Returns copy of args without the command or its arguments.
def updateEnv(env, args):
    arg_idx_next = len(args)
    unset_next_env_var = False
    for arg_idx, arg in enumerate(args[1:]):
        # Support for the -u flag (unsetting) for env command
        # e.g., env -u FOO -u BAR will remove both FOO and BAR
        # from the environment.
        if arg == "-u":
            unset_next_env_var = True
            continue
        # Support for the -i flag which clears the environment
        if arg == "-i":
            env.env = {}
            continue
        if unset_next_env_var:
            unset_next_env_var = False
            if arg in env.env:
                del env.env[arg]
            continue

        # Partition the string into KEY=VALUE.
        key, eq, val = arg.partition("=")
        # Stop if there was no equals.
        if eq == "":
            arg_idx_next = arg_idx + 1
            break
        env.env[key] = val
    return args[arg_idx_next:]


def processRedirects(cmd, stdin_source, cmd_shenv, opened_files):
    """Return the standard fds for cmd after applying redirects

    Returns the three standard file descriptors for the new child process.  Each
    fd may be an open, writable file object or a sentinel value from the
    subprocess module.
    """

    # Apply the redirections, we use (N,) as a sentinel to indicate stdin,
    # stdout, stderr for N equal to 0, 1, or 2 respectively. Redirects to or
    # from a file are represented with a list [file, mode, file-object]
    # where file-object is initially None.
    redirects = [(0,), (1,), (2,)]
    for op, filename in cmd.redirects:
        if op == (">", 2):
            redirects[2] = [filename, "w", None]
        elif op == (">>", 2):
            redirects[2] = [filename, "a", None]
        elif op == (">&", 2) and filename in "012":
            redirects[2] = redirects[int(filename)]
        elif op == (">&",) or op == ("&>",):
            redirects[1] = redirects[2] = [filename, "w", None]
        elif op == (">",):
            redirects[1] = [filename, "w", None]
        elif op == (">>",):
            redirects[1] = [filename, "a", None]
        elif op == ("<",):
            redirects[0] = [filename, "r", None]
        else:
            raise InternalShellError(
                cmd, "Unsupported redirect: %r" % ((op, filename),)
            )

    # Open file descriptors in a second pass.
    std_fds = [None, None, None]
    for index, r in enumerate(redirects):
        # Handle the sentinel values for defaults up front.
        if isinstance(r, tuple):
            if r == (0,):
                fd = stdin_source
            elif r == (1,):
                if index == 0:
                    raise InternalShellError(cmd, "Unsupported redirect for stdin")
                elif index == 1:
                    fd = subprocess.PIPE
                else:
                    fd = subprocess.STDOUT
            elif r == (2,):
                if index != 2:
                    raise InternalShellError(cmd, "Unsupported redirect on stdout")
                fd = subprocess.PIPE
            else:
                raise InternalShellError(cmd, "Bad redirect")
            std_fds[index] = fd
            continue

        filename, mode, fd = r

        # Check if we already have an open fd. This can happen if stdout and
        # stderr go to the same place.
        if fd is not None:
            std_fds[index] = fd
            continue

        redir_filename = None
        name = expand_glob(filename, cmd_shenv.cwd)
        if len(name) != 1:
            raise InternalShellError(
                cmd, "Unsupported: glob in " "redirect expanded to multiple files"
            )
        name = name[0]
        if kAvoidDevNull and name == kDevNull:
            fd = tempfile.TemporaryFile(mode=mode)
        elif kIsWindows and name == "/dev/tty":
            # Simulate /dev/tty on Windows.
            # "CON" is a special filename for the console.
            fd = open("CON", mode)
        else:
            # Make sure relative paths are relative to the cwd.
            redir_filename = os.path.join(cmd_shenv.cwd, name)
            fd = open(redir_filename, mode, encoding="utf-8")
        # Workaround a Win32 and/or subprocess bug when appending.
        #
        # FIXME: Actually, this is probably an instance of PR6753.
        if mode == "a":
            fd.seek(0, 2)
        # Mutate the underlying redirect list so that we can redirect stdout
        # and stderr to the same place without opening the file twice.
        r[2] = fd
        opened_files.append((filename, mode, fd) + (redir_filename,))
        std_fds[index] = fd

    return std_fds


def expand_glob(arg, cwd):
    if isinstance(arg, GlobItem):
        return sorted(arg.resolve(cwd))
    return [arg]


def expand_glob_expressions(args, cwd):
    result = [args[0]]
    for arg in args[1:]:
        result.extend(expand_glob(arg, cwd))
    return result


def quote_windows_command(seq):
    r"""
    Reimplement Python's private subprocess.list2cmdline for MSys compatibility

    Based on CPython implementation here:
      https://hg.python.org/cpython/file/849826a900d2/Lib/subprocess.py#l422

    Some core util distributions (MSys) don't tokenize command line arguments
    the same way that MSVC CRT does. Lit rolls its own quoting logic similar to
    the stock CPython logic to paper over these quoting and tokenization rule
    differences.

    We use the same algorithm from MSDN as CPython
    (http://msdn.microsoft.com/en-us/library/17w5ykft.aspx), but we treat more
    characters as needing quoting, such as double quotes themselves, and square
    brackets.

    For MSys based tools, this is very brittle though, because quoting an
    argument makes the MSys based tool unescape backslashes where it shouldn't
    (e.g. "a\b\\c\\\\d" becomes "a\b\c\\d" where it should stay as it was,
    according to regular win32 command line parsing rules).
    """
    result = []
    needquote = False
    for arg in seq:
        bs_buf = []

        # Add a space to separate this argument from the others
        if result:
            result.append(" ")

        # This logic differs from upstream list2cmdline.
        needquote = (
            (" " in arg)
            or ("\t" in arg)
            or ('"' in arg)
            or ("[" in arg)
            or (";" in arg)
            or not arg
        )
        if needquote:
            result.append('"')

        for c in arg:
            if c == "\\":
                # Don't know if we need to double yet.
                bs_buf.append(c)
            elif c == '"':
                # Double backslashes.
                result.append("\\" * len(bs_buf) * 2)
                bs_buf = []
                result.append('\\"')
            else:
                # Normal char
                if bs_buf:
                    result.extend(bs_buf)
                    bs_buf = []
                result.append(c)

        # Add remaining backslashes, if any.
        if bs_buf:
            result.extend(bs_buf)

        if needquote:
            result.extend(bs_buf)
            result.append('"')

    return "".join(result)
