from __future__ import absolute_import
import difflib
import errno
import functools
import itertools
import getopt
import os, signal, subprocess, sys
import re
import stat
import platform
import shutil
import tempfile
import threading

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from lit.ShCommands import GlobItem
import lit.ShUtil as ShUtil
import lit.Test as Test
import lit.util
from lit.util import to_bytes, to_string
from lit.BooleanExpression import BooleanExpression

class InternalShellError(Exception):
    def __init__(self, command, message):
        self.command = command
        self.message = message

kIsWindows = platform.system() == 'Windows'

# Don't use close_fds on Windows.
kUseCloseFDs = not kIsWindows

# Use temporary files to replace /dev/null on Windows.
kAvoidDevNull = kIsWindows

class ShellEnvironment(object):

    """Mutable shell environment containing things like CWD and env vars.

    Environment variables are not implemented, but cwd tracking is.
    """

    def __init__(self, cwd, env):
        self.cwd = cwd
        self.env = dict(env)

class TimeoutHelper(object):
    """
        Object used to helper manage enforcing a timeout in
        _executeShCmd(). It is passed through recursive calls
        to collect processes that have been executed so that when
        the timeout happens they can be killed.
    """
    def __init__(self, timeout):
        self.timeout = timeout
        self._procs = []
        self._timeoutReached = False
        self._doneKillPass = False
        # This lock will be used to protect concurrent access
        # to _procs and _doneKillPass
        self._lock = None
        self._timer = None

    def cancel(self):
        if not self.active():
            return
        self._timer.cancel()

    def active(self):
        return self.timeout > 0

    def addProcess(self, proc):
        if not self.active():
            return
        needToRunKill = False
        with self._lock:
            self._procs.append(proc)
            # Avoid re-entering the lock by finding out if kill needs to be run
            # again here but call it if necessary once we have left the lock.
            # We could use a reentrant lock here instead but this code seems
            # clearer to me.
            needToRunKill = self._doneKillPass

        # The initial call to _kill() from the timer thread already happened so
        # we need to call it again from this thread, otherwise this process
        # will be left to run even though the timeout was already hit
        if needToRunKill:
            assert self.timeoutReached()
            self._kill()

    def startTimer(self):
        if not self.active():
            return

        # Do some late initialisation that's only needed
        # if there is a timeout set
        self._lock = threading.Lock()
        self._timer = threading.Timer(self.timeout, self._handleTimeoutReached)
        self._timer.start()

    def _handleTimeoutReached(self):
        self._timeoutReached = True
        self._kill()

    def timeoutReached(self):
        return self._timeoutReached

    def _kill(self):
        """
            This method may be called multiple times as we might get unlucky
            and be in the middle of creating a new process in _executeShCmd()
            which won't yet be in ``self._procs``. By locking here and in
            addProcess() we should be able to kill processes launched after
            the initial call to _kill()
        """
        with self._lock:
            for p in self._procs:
                lit.util.killProcessAndChildren(p.pid)
            # Empty the list and note that we've done a pass over the list
            self._procs = [] # Python2 doesn't have list.clear()
            self._doneKillPass = True

class ShellCommandResult(object):
    """Captures the result of an individual command."""

    def __init__(self, command, stdout, stderr, exitCode, timeoutReached,
                 outputFiles = []):
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.exitCode = exitCode
        self.timeoutReached = timeoutReached
        self.outputFiles = list(outputFiles)
               
def executeShCmd(cmd, shenv, results, timeout=0):
    """
        Wrapper around _executeShCmd that handles
        timeout
    """
    # Use the helper even when no timeout is required to make
    # other code simpler (i.e. avoid bunch of ``!= None`` checks)
    timeoutHelper = TimeoutHelper(timeout)
    if timeout > 0:
        timeoutHelper.startTimer()
    finalExitCode = _executeShCmd(cmd, shenv, results, timeoutHelper)
    timeoutHelper.cancel()
    timeoutInfo = None
    if timeoutHelper.timeoutReached():
        timeoutInfo = 'Reached timeout of {} seconds'.format(timeout)

    return (finalExitCode, timeoutInfo)

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
    """
    Reimplement Python's private subprocess.list2cmdline for MSys compatibility

    Based on CPython implementation here:
      https://hg.python.org/cpython/file/849826a900d2/Lib/subprocess.py#l422

    Some core util distributions (MSys) don't tokenize command line arguments
    the same way that MSVC CRT does. Lit rolls its own quoting logic similar to
    the stock CPython logic to paper over these quoting and tokenization rule
    differences.

    We use the same algorithm from MSDN as CPython
    (http://msdn.microsoft.com/en-us/library/17w5ykft.aspx), but we treat more
    characters as needing quoting, such as double quotes themselves.
    """
    result = []
    needquote = False
    for arg in seq:
        bs_buf = []

        # Add a space to separate this argument from the others
        if result:
            result.append(' ')

        # This logic differs from upstream list2cmdline.
        needquote = (" " in arg) or ("\t" in arg) or ("\"" in arg) or not arg
        if needquote:
            result.append('"')

        for c in arg:
            if c == '\\':
                # Don't know if we need to double yet.
                bs_buf.append(c)
            elif c == '"':
                # Double backslashes.
                result.append('\\' * len(bs_buf)*2)
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

    return ''.join(result)

# cmd is export or env
def updateEnv(env, cmd):
    arg_idx = 1
    unset_next_env_var = False
    for arg_idx, arg in enumerate(cmd.args[1:]):
        # Support for the -u flag (unsetting) for env command
        # e.g., env -u FOO -u BAR will remove both FOO and BAR
        # from the environment.
        if arg == '-u':
            unset_next_env_var = True
            continue
        if unset_next_env_var:
            unset_next_env_var = False
            if arg in env.env:
                del env.env[arg]
            continue

        # Partition the string into KEY=VALUE.
        key, eq, val = arg.partition('=')
        # Stop if there was no equals.
        if eq == '':
            break
        env.env[key] = val
    cmd.args = cmd.args[arg_idx+1:]

def executeBuiltinEcho(cmd, shenv):
    """Interpret a redirected echo command"""
    opened_files = []
    stdin, stdout, stderr = processRedirects(cmd, subprocess.PIPE, shenv,
                                             opened_files)
    if stdin != subprocess.PIPE or stderr != subprocess.PIPE:
        raise InternalShellError(
                cmd, "stdin and stderr redirects not supported for echo")

    # Some tests have un-redirected echo commands to help debug test failures.
    # Buffer our output and return it to the caller.
    is_redirected = True
    encode = lambda x : x
    if stdout == subprocess.PIPE:
        is_redirected = False
        stdout = StringIO()
    elif kIsWindows:
        # Reopen stdout in binary mode to avoid CRLF translation. The versions
        # of echo we are replacing on Windows all emit plain LF, and the LLVM
        # tests now depend on this.
        # When we open as binary, however, this also means that we have to write
        # 'bytes' objects to stdout instead of 'str' objects.
        encode = lit.util.to_bytes
        stdout = open(stdout.name, stdout.mode + 'b')
        opened_files.append((None, None, stdout, None))

    # Implement echo flags. We only support -e and -n, and not yet in
    # combination. We have to ignore unknown flags, because `echo "-D FOO"`
    # prints the dash.
    args = cmd.args[1:]
    interpret_escapes = False
    write_newline = True
    while len(args) >= 1 and args[0] in ('-e', '-n'):
        flag = args[0]
        args = args[1:]
        if flag == '-e':
            interpret_escapes = True
        elif flag == '-n':
            write_newline = False

    def maybeUnescape(arg):
        if not interpret_escapes:
            return arg

        arg = lit.util.to_bytes(arg)
        codec = 'string_escape' if sys.version_info < (3,0) else 'unicode_escape'
        return arg.decode(codec)

    if args:
        for arg in args[:-1]:
            stdout.write(encode(maybeUnescape(arg)))
            stdout.write(encode(' '))
        stdout.write(encode(maybeUnescape(args[-1])))
    if write_newline:
        stdout.write(encode('\n'))

    for (name, mode, f, path) in opened_files:
        f.close()

    if not is_redirected:
        return stdout.getvalue()
    return ""

def executeBuiltinMkdir(cmd, cmd_shenv):
    """executeBuiltinMkdir - Create new directories."""
    args = expand_glob_expressions(cmd.args, cmd_shenv.cwd)[1:]
    try:
        opts, args = getopt.gnu_getopt(args, 'p')
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
        if not os.path.isabs(dir):
            dir = os.path.realpath(os.path.join(cmd_shenv.cwd, dir))
        if parent:
            lit.util.mkdir_p(dir)
        else:
            try:
                os.mkdir(dir)
            except OSError as err:
                stderr.write("Error: 'mkdir' command failed, %s\n" % str(err))
                exitCode = 1
    return ShellCommandResult(cmd, "", stderr.getvalue(), exitCode, False)

def executeBuiltinDiff(cmd, cmd_shenv):
    """executeBuiltinDiff - Compare files line by line."""
    args = expand_glob_expressions(cmd.args, cmd_shenv.cwd)[1:]
    try:
        opts, args = getopt.gnu_getopt(args, "wbur", ["strip-trailing-cr"])
    except getopt.GetoptError as err:
        raise InternalShellError(cmd, "Unsupported: 'diff':  %s" % str(err))

    filelines, filepaths, dir_trees = ([] for i in range(3))
    ignore_all_space = False
    ignore_space_change = False
    unified_diff = False
    recursive_diff = False
    strip_trailing_cr = False
    for o, a in opts:
        if o == "-w":
            ignore_all_space = True
        elif o == "-b":
            ignore_space_change = True
        elif o == "-u":
            unified_diff = True
        elif o == "-r":
            recursive_diff = True
        elif o == "--strip-trailing-cr":
            strip_trailing_cr = True
        else:
            assert False, "unhandled option"

    if len(args) != 2:
        raise InternalShellError(cmd, "Error:  missing or extra operand")

    def getDirTree(path, basedir=""):
        # Tree is a tuple of form (dirname, child_trees).
        # An empty dir has child_trees = [], a file has child_trees = None.
        child_trees = []
        for dirname, child_dirs, files in os.walk(os.path.join(basedir, path)):
            for child_dir in child_dirs:
                child_trees.append(getDirTree(child_dir, dirname))
            for filename in files:
                child_trees.append((filename, None))
            return path, sorted(child_trees)

    def compareTwoFiles(filepaths):
        filelines = []
        for file in filepaths:
            with open(file, 'r') as f:
                filelines.append(f.readlines())

        exitCode = 0 
        def compose2(f, g):
            return lambda x: f(g(x))

        f = lambda x: x
        if strip_trailing_cr:
            f = compose2(lambda line: line.rstrip('\r'), f)
        if ignore_all_space or ignore_space_change:
            ignoreSpace = lambda line, separator: separator.join(line.split())
            ignoreAllSpaceOrSpaceChange = functools.partial(ignoreSpace, separator='' if ignore_all_space else ' ')
            f = compose2(ignoreAllSpaceOrSpaceChange, f)

        for idx, lines in enumerate(filelines):
            filelines[idx]= [f(line) for line in lines]

        func = difflib.unified_diff if unified_diff else difflib.context_diff
        for diff in func(filelines[0], filelines[1], filepaths[0], filepaths[1]):
            stdout.write(diff)
            exitCode = 1
        return exitCode

    def printDirVsFile(dir_path, file_path):
        if os.path.getsize(file_path):
            msg = "File %s is a directory while file %s is a regular file"
        else:
            msg = "File %s is a directory while file %s is a regular empty file"
        stdout.write(msg % (dir_path, file_path) + "\n")

    def printFileVsDir(file_path, dir_path):
        if os.path.getsize(file_path):
            msg = "File %s is a regular file while file %s is a directory"
        else:
            msg = "File %s is a regular empty file while file %s is a directory"
        stdout.write(msg % (file_path, dir_path) + "\n")

    def printOnlyIn(basedir, path, name):
        stdout.write("Only in %s: %s\n" % (os.path.join(basedir, path), name))

    def compareDirTrees(dir_trees, base_paths=["", ""]):
        # Dirnames of the trees are not checked, it's caller's responsibility,
        # as top-level dirnames are always different. Base paths are important
        # for doing os.walk, but we don't put it into tree's dirname in order
        # to speed up string comparison below and while sorting in getDirTree.
        left_tree, right_tree = dir_trees[0], dir_trees[1]
        left_base, right_base = base_paths[0], base_paths[1]

        # Compare two files or report file vs. directory mismatch.
        if left_tree[1] is None and right_tree[1] is None:
            return compareTwoFiles([os.path.join(left_base, left_tree[0]),
                                    os.path.join(right_base, right_tree[0])])

        if left_tree[1] is None and right_tree[1] is not None:
            printFileVsDir(os.path.join(left_base, left_tree[0]),
                           os.path.join(right_base, right_tree[0]))
            return 1

        if left_tree[1] is not None and right_tree[1] is None:
            printDirVsFile(os.path.join(left_base, left_tree[0]),
                           os.path.join(right_base, right_tree[0]))
            return 1

        # Compare two directories via recursive use of compareDirTrees.
        exitCode = 0
        left_names = [node[0] for node in left_tree[1]]
        right_names = [node[0] for node in right_tree[1]]
        l, r = 0, 0
        while l < len(left_names) and r < len(right_names):
            # Names are sorted in getDirTree, rely on that order.
            if left_names[l] < right_names[r]:
                exitCode = 1
                printOnlyIn(left_base, left_tree[0], left_names[l])
                l += 1
            elif left_names[l] > right_names[r]:
                exitCode = 1
                printOnlyIn(right_base, right_tree[0], right_names[r])
                r += 1
            else:
                exitCode |= compareDirTrees([left_tree[1][l], right_tree[1][r]],
                                            [os.path.join(left_base, left_tree[0]),
                                            os.path.join(right_base, right_tree[0])])
                l += 1
                r += 1

        # At least one of the trees has ended. Report names from the other tree.
        while l < len(left_names):
            exitCode = 1
            printOnlyIn(left_base, left_tree[0], left_names[l])
            l += 1
        while r < len(right_names):
            exitCode = 1
            printOnlyIn(right_base, right_tree[0], right_names[r])
            r += 1
        return exitCode

    stderr = StringIO()
    stdout = StringIO()
    exitCode = 0
    try:
        for file in args:
            if not os.path.isabs(file):
                file = os.path.realpath(os.path.join(cmd_shenv.cwd, file))
    
            if recursive_diff:
                dir_trees.append(getDirTree(file))
            else:
                filepaths.append(file)

        if not recursive_diff:
            exitCode = compareTwoFiles(filepaths)
        else:
            exitCode = compareDirTrees(dir_trees)

    except IOError as err:
        stderr.write("Error: 'diff' command failed, %s\n" % str(err))
        exitCode = 1

    return ShellCommandResult(cmd, stdout.getvalue(), stderr.getvalue(), exitCode, False)

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
        os.chmod(path, stat.S_IMODE( os.stat(path).st_mode) | stat.S_IWRITE)
        os.remove(path)

    stderr = StringIO()
    exitCode = 0
    for path in args:
        if not os.path.isabs(path):
            path = os.path.realpath(os.path.join(cmd_shenv.cwd, path))
        if force and not os.path.exists(path):
            continue
        try:
            if os.path.isdir(path):
                if not recursive:
                    stderr.write("Error: %s is a directory\n" % path)
                    exitCode = 1
                shutil.rmtree(path, onerror = on_rm_error if force else None)
            else:
                if force and not os.access(path, os.W_OK):
                    os.chmod(path,
                             stat.S_IMODE(os.stat(path).st_mode) | stat.S_IWRITE)
                os.remove(path)
        except OSError as err:
            stderr.write("Error: 'rm' command failed, %s" % str(err))
            exitCode = 1
    return ShellCommandResult(cmd, "", stderr.getvalue(), exitCode, False)

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
    for (op, filename) in cmd.redirects:
        if op == ('>',2):
            redirects[2] = [filename, 'w', None]
        elif op == ('>>',2):
            redirects[2] = [filename, 'a', None]
        elif op == ('>&',2) and filename in '012':
            redirects[2] = redirects[int(filename)]
        elif op == ('>&',) or op == ('&>',):
            redirects[1] = redirects[2] = [filename, 'w', None]
        elif op == ('>',):
            redirects[1] = [filename, 'w', None]
        elif op == ('>>',):
            redirects[1] = [filename, 'a', None]
        elif op == ('<',):
            redirects[0] = [filename, 'r', None]
        else:
            raise InternalShellError(cmd, "Unsupported redirect: %r" % ((op, filename),))

    # Open file descriptors in a second pass.
    std_fds = [None, None, None]
    for (index, r) in enumerate(redirects):
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

        (filename, mode, fd) = r

        # Check if we already have an open fd. This can happen if stdout and
        # stderr go to the same place.
        if fd is not None:
            std_fds[index] = fd
            continue

        redir_filename = None
        name = expand_glob(filename, cmd_shenv.cwd)
        if len(name) != 1:
           raise InternalShellError(cmd, "Unsupported: glob in "
                                    "redirect expanded to multiple files")
        name = name[0]
        if kAvoidDevNull and name == '/dev/null':
            fd = tempfile.TemporaryFile(mode=mode)
        elif kIsWindows and name == '/dev/tty':
            # Simulate /dev/tty on Windows.
            # "CON" is a special filename for the console.
            fd = open("CON", mode)
        else:
            # Make sure relative paths are relative to the cwd.
            redir_filename = os.path.join(cmd_shenv.cwd, name)
            fd = open(redir_filename, mode)
        # Workaround a Win32 and/or subprocess bug when appending.
        #
        # FIXME: Actually, this is probably an instance of PR6753.
        if mode == 'a':
            fd.seek(0, 2)
        # Mutate the underlying redirect list so that we can redirect stdout
        # and stderr to the same place without opening the file twice.
        r[2] = fd
        opened_files.append((filename, mode, fd) + (redir_filename,))
        std_fds[index] = fd

    return std_fds

def _executeShCmd(cmd, shenv, results, timeoutHelper):
    if timeoutHelper.timeoutReached():
        # Prevent further recursion if the timeout has been hit
        # as we should try avoid launching more processes.
        return None

    if isinstance(cmd, ShUtil.Seq):
        if cmd.op == ';':
            res = _executeShCmd(cmd.lhs, shenv, results, timeoutHelper)
            return _executeShCmd(cmd.rhs, shenv, results, timeoutHelper)

        if cmd.op == '&':
            raise InternalShellError(cmd,"unsupported shell operator: '&'")

        if cmd.op == '||':
            res = _executeShCmd(cmd.lhs, shenv, results, timeoutHelper)
            if res != 0:
                res = _executeShCmd(cmd.rhs, shenv, results, timeoutHelper)
            return res

        if cmd.op == '&&':
            res = _executeShCmd(cmd.lhs, shenv, results, timeoutHelper)
            if res is None:
                return res

            if res == 0:
                res = _executeShCmd(cmd.rhs, shenv, results, timeoutHelper)
            return res

        raise ValueError('Unknown shell command: %r' % cmd.op)
    assert isinstance(cmd, ShUtil.Pipeline)

    # Handle shell builtins first.
    if cmd.commands[0].args[0] == 'cd':
        if len(cmd.commands) != 1:
            raise ValueError("'cd' cannot be part of a pipeline")
        if len(cmd.commands[0].args) != 2:
            raise ValueError("'cd' supports only one argument")
        newdir = cmd.commands[0].args[1]
        # Update the cwd in the parent environment.
        if os.path.isabs(newdir):
            shenv.cwd = newdir
        else:
            shenv.cwd = os.path.realpath(os.path.join(shenv.cwd, newdir))
        # The cd builtin always succeeds. If the directory does not exist, the
        # following Popen calls will fail instead.
        return 0

    # Handle "echo" as a builtin if it is not part of a pipeline. This greatly
    # speeds up tests that construct input files by repeatedly echo-appending to
    # a file.
    # FIXME: Standardize on the builtin echo implementation. We can use a
    # temporary file to sidestep blocking pipe write issues.
    if cmd.commands[0].args[0] == 'echo' and len(cmd.commands) == 1:
        output = executeBuiltinEcho(cmd.commands[0], shenv)
        results.append(ShellCommandResult(cmd.commands[0], output, "", 0,
                                          False))
        return 0

    if cmd.commands[0].args[0] == 'export':
        if len(cmd.commands) != 1:
            raise ValueError("'export' cannot be part of a pipeline")
        if len(cmd.commands[0].args) != 2:
            raise ValueError("'export' supports only one argument")
        updateEnv(shenv, cmd.commands[0])
        return 0

    if cmd.commands[0].args[0] == 'mkdir':
        if len(cmd.commands) != 1:
            raise InternalShellError(cmd.commands[0], "Unsupported: 'mkdir' "
                                     "cannot be part of a pipeline")
        cmdResult = executeBuiltinMkdir(cmd.commands[0], shenv)
        results.append(cmdResult)
        return cmdResult.exitCode

    if cmd.commands[0].args[0] == 'diff':
        if len(cmd.commands) != 1:
            raise InternalShellError(cmd.commands[0], "Unsupported: 'diff' "
                                     "cannot be part of a pipeline")
        cmdResult = executeBuiltinDiff(cmd.commands[0], shenv)
        results.append(cmdResult)
        return cmdResult.exitCode

    if cmd.commands[0].args[0] == 'rm':
        if len(cmd.commands) != 1:
            raise InternalShellError(cmd.commands[0], "Unsupported: 'rm' "
                                     "cannot be part of a pipeline")
        cmdResult = executeBuiltinRm(cmd.commands[0], shenv)
        results.append(cmdResult)
        return cmdResult.exitCode

    procs = []
    default_stdin = subprocess.PIPE
    stderrTempFiles = []
    opened_files = []
    named_temp_files = []
    builtin_commands = set(['cat'])
    builtin_commands_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "builtin_commands")
    # To avoid deadlock, we use a single stderr stream for piped
    # output. This is null until we have seen some output using
    # stderr.
    for i,j in enumerate(cmd.commands):
        # Reference the global environment by default.
        cmd_shenv = shenv
        if j.args[0] == 'env':
            # Create a copy of the global environment and modify it for this one
            # command. There might be multiple envs in a pipeline:
            #   env FOO=1 llc < %s | env BAR=2 llvm-mc | FileCheck %s
            cmd_shenv = ShellEnvironment(shenv.cwd, shenv.env)
            updateEnv(cmd_shenv, j)

        stdin, stdout, stderr = processRedirects(j, default_stdin, cmd_shenv,
                                                 opened_files)

        # If stderr wants to come from stdout, but stdout isn't a pipe, then put
        # stderr on a pipe and treat it as stdout.
        if (stderr == subprocess.STDOUT and stdout != subprocess.PIPE):
            stderr = subprocess.PIPE
            stderrIsStdout = True
        else:
            stderrIsStdout = False

            # Don't allow stderr on a PIPE except for the last
            # process, this could deadlock.
            #
            # FIXME: This is slow, but so is deadlock.
            if stderr == subprocess.PIPE and j != cmd.commands[-1]:
                stderr = tempfile.TemporaryFile(mode='w+b')
                stderrTempFiles.append((i, stderr))

        # Resolve the executable path ourselves.
        args = list(j.args)
        executable = None
        is_builtin_cmd = args[0] in builtin_commands;
        if not is_builtin_cmd:
            # For paths relative to cwd, use the cwd of the shell environment.
            if args[0].startswith('.'):
                exe_in_cwd = os.path.join(cmd_shenv.cwd, args[0])
                if os.path.isfile(exe_in_cwd):
                    executable = exe_in_cwd
            if not executable:
                executable = lit.util.which(args[0], cmd_shenv.env['PATH'])
            if not executable:
                raise InternalShellError(j, '%r: command not found' % j.args[0])

        # Replace uses of /dev/null with temporary files.
        if kAvoidDevNull:
            for i,arg in enumerate(args):
                if arg == "/dev/null":
                    f = tempfile.NamedTemporaryFile(delete=False)
                    f.close()
                    named_temp_files.append(f.name)
                    args[i] = f.name

        # Expand all glob expressions
        args = expand_glob_expressions(args, cmd_shenv.cwd)
        if is_builtin_cmd:
            args.insert(0, "python")
            args[1] = os.path.join(builtin_commands_dir ,args[1] + ".py")

        # On Windows, do our own command line quoting for better compatibility
        # with some core utility distributions.
        if kIsWindows:
            args = quote_windows_command(args)

        try:
            procs.append(subprocess.Popen(args, cwd=cmd_shenv.cwd,
                                          executable = executable,
                                          stdin = stdin,
                                          stdout = stdout,
                                          stderr = stderr,
                                          env = cmd_shenv.env,
                                          close_fds = kUseCloseFDs))
            # Let the helper know about this process
            timeoutHelper.addProcess(procs[-1])
        except OSError as e:
            raise InternalShellError(j, 'Could not create process ({}) due to {}'.format(executable, e))

        # Immediately close stdin for any process taking stdin from us.
        if stdin == subprocess.PIPE:
            procs[-1].stdin.close()
            procs[-1].stdin = None

        # Update the current stdin source.
        if stdout == subprocess.PIPE:
            default_stdin = procs[-1].stdout
        elif stderrIsStdout:
            default_stdin = procs[-1].stderr
        else:
            default_stdin = subprocess.PIPE

    # Explicitly close any redirected files. We need to do this now because we
    # need to release any handles we may have on the temporary files (important
    # on Win32, for example). Since we have already spawned the subprocess, our
    # handles have already been transferred so we do not need them anymore.
    for (name, mode, f, path) in opened_files:
        f.close()

    # FIXME: There is probably still deadlock potential here. Yawn.
    procData = [None] * len(procs)
    procData[-1] = procs[-1].communicate()

    for i in range(len(procs) - 1):
        if procs[i].stdout is not None:
            out = procs[i].stdout.read()
        else:
            out = ''
        if procs[i].stderr is not None:
            err = procs[i].stderr.read()
        else:
            err = ''
        procData[i] = (out,err)

    # Read stderr out of the temp files.
    for i,f in stderrTempFiles:
        f.seek(0, 0)
        procData[i] = (procData[i][0], f.read())

    def to_string(bytes):
        if isinstance(bytes, str):
            return bytes
        return bytes.encode('utf-8')

    exitCode = None
    for i,(out,err) in enumerate(procData):
        res = procs[i].wait()
        # Detect Ctrl-C in subprocess.
        if res == -signal.SIGINT:
            raise KeyboardInterrupt

        # Ensure the resulting output is always of string type.
        try:
            if out is None:
                out = ''
            else:
                out = to_string(out.decode('utf-8', errors='replace'))
        except:
            out = str(out)
        try:
            if err is None:
                err = ''
            else:
                err = to_string(err.decode('utf-8', errors='replace'))
        except:
            err = str(err)

        # Gather the redirected output files for failed commands.
        output_files = []
        if res != 0:
            for (name, mode, f, path) in sorted(opened_files):
                if path is not None and mode in ('w', 'a'):
                    try:
                        with open(path, 'rb') as f:
                            data = f.read()
                    except:
                        data = None
                    if data is not None:
                        output_files.append((name, path, data))
            
        results.append(ShellCommandResult(
            cmd.commands[i], out, err, res, timeoutHelper.timeoutReached(),
            output_files))
        if cmd.pipe_err:
            # Take the last failing exit code from the pipeline.
            if not exitCode or res != 0:
                exitCode = res
        else:
            exitCode = res

    # Remove any named temporary files we created.
    for f in named_temp_files:
        try:
            os.remove(f)
        except OSError:
            pass

    if cmd.negate:
        exitCode = not exitCode

    return exitCode

def executeScriptInternal(test, litConfig, tmpBase, commands, cwd):
    cmds = []
    for ln in commands:
        try:
            cmds.append(ShUtil.ShParser(ln, litConfig.isWindows,
                                        test.config.pipefail).parse())
        except:
            return lit.Test.Result(Test.FAIL, "shell parser error on: %r" % ln)

    cmd = cmds[0]
    for c in cmds[1:]:
        cmd = ShUtil.Seq(cmd, '&&', c)

    results = []
    timeoutInfo = None
    try:
        shenv = ShellEnvironment(cwd, test.config.environment)
        exitCode, timeoutInfo = executeShCmd(cmd, shenv, results, timeout=litConfig.maxIndividualTestTime)
    except InternalShellError:
        e = sys.exc_info()[1]
        exitCode = 127
        results.append(
            ShellCommandResult(e.command, '', e.message, exitCode, False))

    out = err = ''
    for i,result in enumerate(results):
        # Write the command line run.
        out += '$ %s\n' % (' '.join('"%s"' % s
                                    for s in result.command.args),)

        # If nothing interesting happened, move on.
        if litConfig.maxIndividualTestTime == 0 and \
               result.exitCode == 0 and \
               not result.stdout.strip() and not result.stderr.strip():
            continue

        # Otherwise, something failed or was printed, show it.

        # Add the command output, if redirected.
        for (name, path, data) in result.outputFiles:
            if data.strip():
                out += "# redirected output from %r:\n" % (name,)
                data = to_string(data.decode('utf-8', errors='replace'))
                if len(data) > 1024:
                    out += data[:1024] + "\n...\n"
                    out += "note: data was truncated\n"
                else:
                    out += data
                out += "\n"
                    
        if result.stdout.strip():
            out += '# command output:\n%s\n' % (result.stdout,)
        if result.stderr.strip():
            out += '# command stderr:\n%s\n' % (result.stderr,)
        if not result.stdout.strip() and not result.stderr.strip():
            out += "note: command had no output on stdout or stderr\n"

        # Show the error conditions:
        if result.exitCode != 0:
            # On Windows, a negative exit code indicates a signal, and those are
            # easier to recognize or look up if we print them in hex.
            if litConfig.isWindows and result.exitCode < 0:
                codeStr = hex(int(result.exitCode & 0xFFFFFFFF)).rstrip("L")
            else:
                codeStr = str(result.exitCode)
            out += "error: command failed with exit status: %s\n" % (
                codeStr,)
        if litConfig.maxIndividualTestTime > 0:
            out += 'error: command reached timeout: %s\n' % (
                str(result.timeoutReached),)

    return out, err, exitCode, timeoutInfo

def executeScript(test, litConfig, tmpBase, commands, cwd):
    bashPath = litConfig.getBashPath()
    isWin32CMDEXE = (litConfig.isWindows and not bashPath)
    script = tmpBase + '.script'
    if isWin32CMDEXE:
        script += '.bat'

    # Write script file
    mode = 'w'
    if litConfig.isWindows and not isWin32CMDEXE:
      mode += 'b'  # Avoid CRLFs when writing bash scripts.
    f = open(script, mode)
    if isWin32CMDEXE:
        f.write('@echo off\n')
        f.write('\nif %ERRORLEVEL% NEQ 0 EXIT\n'.join(commands))
    else:
        if test.config.pipefail:
            f.write('set -o pipefail;')
        if litConfig.echo_all_commands:
            f.write('set -x;')
        f.write('{ ' + '; } &&\n{ '.join(commands) + '; }')
    f.write('\n')
    f.close()

    if isWin32CMDEXE:
        command = ['cmd','/c', script]
    else:
        if bashPath:
            command = [bashPath, script]
        else:
            command = ['/bin/sh', script]
        if litConfig.useValgrind:
            # FIXME: Running valgrind on sh is overkill. We probably could just
            # run on clang with no real loss.
            command = litConfig.valgrindArgs + command

    try:
        out, err, exitCode = lit.util.executeCommand(command, cwd=cwd,
                                       env=test.config.environment,
                                       timeout=litConfig.maxIndividualTestTime)
        return (out, err, exitCode, None)
    except lit.util.ExecuteCommandTimeoutException as e:
        return (e.out, e.err, e.exitCode, e.msg)

def parseIntegratedTestScriptCommands(source_path, keywords):
    """
    parseIntegratedTestScriptCommands(source_path) -> commands

    Parse the commands in an integrated test script file into a list of
    (line_number, command_type, line).
    """

    # This code is carefully written to be dual compatible with Python 2.5+ and
    # Python 3 without requiring input files to always have valid codings. The
    # trick we use is to open the file in binary mode and use the regular
    # expression library to find the commands, with it scanning strings in
    # Python2 and bytes in Python3.
    #
    # Once we find a match, we do require each script line to be decodable to
    # UTF-8, so we convert the outputs to UTF-8 before returning. This way the
    # remaining code can work with "strings" agnostic of the executing Python
    # version.

    keywords_re = re.compile(
        to_bytes("(%s)(.*)\n" % ("|".join(re.escape(k) for k in keywords),)))

    f = open(source_path, 'rb')
    try:
        # Read the entire file contents.
        data = f.read()

        # Ensure the data ends with a newline.
        if not data.endswith(to_bytes('\n')):
            data = data + to_bytes('\n')

        # Iterate over the matches.
        line_number = 1
        last_match_position = 0
        for match in keywords_re.finditer(data):
            # Compute the updated line number by counting the intervening
            # newlines.
            match_position = match.start()
            line_number += data.count(to_bytes('\n'), last_match_position,
                                      match_position)
            last_match_position = match_position

            # Convert the keyword and line to UTF-8 strings and yield the
            # command. Note that we take care to return regular strings in
            # Python 2, to avoid other code having to differentiate between the
            # str and unicode types.
            #
            # Opening the file in binary mode prevented Windows \r newline
            # characters from being converted to Unix \n newlines, so manually
            # strip those from the yielded lines.
            keyword,ln = match.groups()
            yield (line_number, to_string(keyword.decode('utf-8')),
                   to_string(ln.decode('utf-8').rstrip('\r')))
    finally:
        f.close()

def getTempPaths(test):
    """Get the temporary location, this is always relative to the test suite
    root, not test source root."""
    execpath = test.getExecPath()
    execdir,execbase = os.path.split(execpath)
    tmpDir = os.path.join(execdir, 'Output')
    tmpBase = os.path.join(tmpDir, execbase)
    return tmpDir, tmpBase

def colonNormalizePath(path):
    if kIsWindows:
        return re.sub(r'^(.):', r'\1', path.replace('\\', '/'))
    else:
        assert path[0] == '/'
        return path[1:]

def getDefaultSubstitutions(test, tmpDir, tmpBase, normalize_slashes=False):
    sourcepath = test.getSourcePath()
    sourcedir = os.path.dirname(sourcepath)

    # Normalize slashes, if requested.
    if normalize_slashes:
        sourcepath = sourcepath.replace('\\', '/')
        sourcedir = sourcedir.replace('\\', '/')
        tmpDir = tmpDir.replace('\\', '/')
        tmpBase = tmpBase.replace('\\', '/')

    # We use #_MARKER_# to hide %% while we do the other substitutions.
    substitutions = []
    substitutions.extend([('%%', '#_MARKER_#')])
    substitutions.extend(test.config.substitutions)
    tmpName = tmpBase + '.tmp'
    baseName = os.path.basename(tmpBase)
    substitutions.extend([('%s', sourcepath),
                          ('%S', sourcedir),
                          ('%p', sourcedir),
                          ('%{pathsep}', os.pathsep),
                          ('%t', tmpName),
                          ('%basename_t', baseName),
                          ('%T', tmpDir),
                          ('#_MARKER_#', '%')])

    # "%/[STpst]" should be normalized.
    substitutions.extend([
            ('%/s', sourcepath.replace('\\', '/')),
            ('%/S', sourcedir.replace('\\', '/')),
            ('%/p', sourcedir.replace('\\', '/')),
            ('%/t', tmpBase.replace('\\', '/') + '.tmp'),
            ('%/T', tmpDir.replace('\\', '/')),
            ])

    # "%:[STpst]" are normalized paths without colons and without a leading
    # slash.
    substitutions.extend([
            ('%:s', colonNormalizePath(sourcepath)),
            ('%:S', colonNormalizePath(sourcedir)),
            ('%:p', colonNormalizePath(sourcedir)),
            ('%:t', colonNormalizePath(tmpBase + '.tmp')),
            ('%:T', colonNormalizePath(tmpDir)),
            ])
    return substitutions

def applySubstitutions(script, substitutions):
    """Apply substitutions to the script.  Allow full regular expression syntax.
    Replace each matching occurrence of regular expression pattern a with
    substitution b in line ln."""
    def processLine(ln):
        # Apply substitutions
        for a,b in substitutions:
            if kIsWindows:
                b = b.replace("\\","\\\\")
            ln = re.sub(a, b, ln)

        # Strip the trailing newline and any extra whitespace.
        return ln.strip()
    # Note Python 3 map() gives an iterator rather than a list so explicitly
    # convert to list before returning.
    return list(map(processLine, script))


class ParserKind(object):
    """
    An enumeration representing the style of an integrated test keyword or
    command.

    TAG: A keyword taking no value. Ex 'END.'
    COMMAND: A keyword taking a list of shell commands. Ex 'RUN:'
    LIST: A keyword taking a comma-separated list of values.
    BOOLEAN_EXPR: A keyword taking a comma-separated list of 
        boolean expressions. Ex 'XFAIL:'
    CUSTOM: A keyword with custom parsing semantics.
    """
    TAG = 0
    COMMAND = 1
    LIST = 2
    BOOLEAN_EXPR = 3
    CUSTOM = 4

    @staticmethod
    def allowedKeywordSuffixes(value):
        return { ParserKind.TAG:          ['.'],
                 ParserKind.COMMAND:      [':'],
                 ParserKind.LIST:         [':'],
                 ParserKind.BOOLEAN_EXPR: [':'],
                 ParserKind.CUSTOM:       [':', '.']
               } [value]

    @staticmethod
    def str(value):
        return { ParserKind.TAG:          'TAG',
                 ParserKind.COMMAND:      'COMMAND',
                 ParserKind.LIST:         'LIST',
                 ParserKind.BOOLEAN_EXPR: 'BOOLEAN_EXPR',
                 ParserKind.CUSTOM:       'CUSTOM'
               } [value]


class IntegratedTestKeywordParser(object):
    """A parser for LLVM/Clang style integrated test scripts.

    keyword: The keyword to parse for. It must end in either '.' or ':'.
    kind: An value of ParserKind.
    parser: A custom parser. This value may only be specified with
            ParserKind.CUSTOM.
    """
    def __init__(self, keyword, kind, parser=None, initial_value=None):
        allowedSuffixes = ParserKind.allowedKeywordSuffixes(kind)
        if len(keyword) == 0 or keyword[-1] not in allowedSuffixes:
            if len(allowedSuffixes) == 1:
                raise ValueError("Keyword '%s' of kind '%s' must end in '%s'"
                                 % (keyword, ParserKind.str(kind),
                                    allowedSuffixes[0]))
            else:
                raise ValueError("Keyword '%s' of kind '%s' must end in "
                                 " one of '%s'"
                                 % (keyword, ParserKind.str(kind),
                                    ' '.join(allowedSuffixes)))

        if parser is not None and kind != ParserKind.CUSTOM:
            raise ValueError("custom parsers can only be specified with "
                             "ParserKind.CUSTOM")
        self.keyword = keyword
        self.kind = kind
        self.parsed_lines = []
        self.value = initial_value
        self.parser = parser

        if kind == ParserKind.COMMAND:
            self.parser = self._handleCommand
        elif kind == ParserKind.LIST:
            self.parser = self._handleList
        elif kind == ParserKind.BOOLEAN_EXPR:
            self.parser = self._handleBooleanExpr
        elif kind == ParserKind.TAG:
            self.parser = self._handleTag
        elif kind == ParserKind.CUSTOM:
            if parser is None:
                raise ValueError("ParserKind.CUSTOM requires a custom parser")
            self.parser = parser
        else:
            raise ValueError("Unknown kind '%s'" % kind)

    def parseLine(self, line_number, line):
        try:
            self.parsed_lines += [(line_number, line)]
            self.value = self.parser(line_number, line, self.value)
        except ValueError as e:
            raise ValueError(str(e) + ("\nin %s directive on test line %d" %
                                       (self.keyword, line_number)))

    def getValue(self):
        return self.value

    @staticmethod
    def _handleTag(line_number, line, output):
        """A helper for parsing TAG type keywords"""
        return (not line.strip() or output)

    @staticmethod
    def _handleCommand(line_number, line, output):
        """A helper for parsing COMMAND type keywords"""
        # Trim trailing whitespace.
        line = line.rstrip()
        # Substitute line number expressions
        line = re.sub('%\(line\)', str(line_number), line)

        def replace_line_number(match):
            if match.group(1) == '+':
                return str(line_number + int(match.group(2)))
            if match.group(1) == '-':
                return str(line_number - int(match.group(2)))
        line = re.sub('%\(line *([\+-]) *(\d+)\)', replace_line_number, line)
        # Collapse lines with trailing '\\'.
        if output and output[-1][-1] == '\\':
            output[-1] = output[-1][:-1] + line
        else:
            if output is None:
                output = []
            output.append(line)
        return output

    @staticmethod
    def _handleList(line_number, line, output):
        """A parser for LIST type keywords"""
        if output is None:
            output = []
        output.extend([s.strip() for s in line.split(',')])
        return output

    @staticmethod
    def _handleBooleanExpr(line_number, line, output):
        """A parser for BOOLEAN_EXPR type keywords"""
        if output is None:
            output = []
        output.extend([s.strip() for s in line.split(',')])
        # Evaluate each expression to verify syntax.
        # We don't want any results, just the raised ValueError.
        for s in output:
            if s != '*':
                BooleanExpression.evaluate(s, [])
        return output

    @staticmethod
    def _handleRequiresAny(line_number, line, output):
        """A custom parser to transform REQUIRES-ANY: into REQUIRES:"""

        # Extract the conditions specified in REQUIRES-ANY: as written.
        conditions = []
        IntegratedTestKeywordParser._handleList(line_number, line, conditions)

        # Output a `REQUIRES: a || b || c` expression in its place.
        expression = ' || '.join(conditions)
        IntegratedTestKeywordParser._handleBooleanExpr(line_number,
                                                       expression, output)
        return output

def parseIntegratedTestScript(test, additional_parsers=[],
                              require_script=True):
    """parseIntegratedTestScript - Scan an LLVM/Clang style integrated test
    script and extract the lines to 'RUN' as well as 'XFAIL' and 'REQUIRES'
    and 'UNSUPPORTED' information.

    If additional parsers are specified then the test is also scanned for the
    keywords they specify and all matches are passed to the custom parser.

    If 'require_script' is False an empty script
    may be returned. This can be used for test formats where the actual script
    is optional or ignored.
    """

    # Install the built-in keyword parsers.
    script = []
    builtin_parsers = [
        IntegratedTestKeywordParser('RUN:', ParserKind.COMMAND,
                                    initial_value=script),
        IntegratedTestKeywordParser('XFAIL:', ParserKind.BOOLEAN_EXPR,
                                    initial_value=test.xfails),
        IntegratedTestKeywordParser('REQUIRES:', ParserKind.BOOLEAN_EXPR,
                                    initial_value=test.requires),
        IntegratedTestKeywordParser('REQUIRES-ANY:', ParserKind.CUSTOM,
                                    IntegratedTestKeywordParser._handleRequiresAny, 
                                    initial_value=test.requires), 
        IntegratedTestKeywordParser('UNSUPPORTED:', ParserKind.BOOLEAN_EXPR,
                                    initial_value=test.unsupported),
        IntegratedTestKeywordParser('END.', ParserKind.TAG)
    ]
    keyword_parsers = {p.keyword: p for p in builtin_parsers}
    
    # Install user-defined additional parsers.
    for parser in additional_parsers:
        if not isinstance(parser, IntegratedTestKeywordParser):
            raise ValueError('additional parser must be an instance of '
                             'IntegratedTestKeywordParser')
        if parser.keyword in keyword_parsers:
            raise ValueError("Parser for keyword '%s' already exists"
                             % parser.keyword)
        keyword_parsers[parser.keyword] = parser
        
    # Collect the test lines from the script.
    sourcepath = test.getSourcePath()
    for line_number, command_type, ln in \
            parseIntegratedTestScriptCommands(sourcepath,
                                              keyword_parsers.keys()):
        parser = keyword_parsers[command_type]
        parser.parseLine(line_number, ln)
        if command_type == 'END.' and parser.getValue() is True:
            break

    # Verify the script contains a run line.
    if require_script and not script:
        return lit.Test.Result(Test.UNRESOLVED, "Test has no run line!")

    # Check for unterminated run lines.
    if script and script[-1][-1] == '\\':
        return lit.Test.Result(Test.UNRESOLVED,
                               "Test has unterminated run lines (with '\\')")

    # Enforce REQUIRES:
    missing_required_features = test.getMissingRequiredFeatures()
    if missing_required_features:
        msg = ', '.join(missing_required_features)
        return lit.Test.Result(Test.UNSUPPORTED,
                               "Test requires the following unavailable "
                               "features: %s" % msg)

    # Enforce UNSUPPORTED:
    unsupported_features = test.getUnsupportedFeatures()
    if unsupported_features:
        msg = ', '.join(unsupported_features)
        return lit.Test.Result(
            Test.UNSUPPORTED,
            "Test does not support the following features "
            "and/or targets: %s" % msg)

    # Enforce limit_to_features.
    if not test.isWithinFeatureLimits():
        msg = ', '.join(test.config.limit_to_features)
        return lit.Test.Result(Test.UNSUPPORTED,
                               "Test does not require any of the features "
                               "specified in limit_to_features: %s" % msg)

    return script


def _runShTest(test, litConfig, useExternalSh, script, tmpBase):
    # Create the output directory if it does not already exist.
    lit.util.mkdir_p(os.path.dirname(tmpBase))

    execdir = os.path.dirname(test.getExecPath())
    if useExternalSh:
        res = executeScript(test, litConfig, tmpBase, script, execdir)
    else:
        res = executeScriptInternal(test, litConfig, tmpBase, script, execdir)
    if isinstance(res, lit.Test.Result):
        return res

    out,err,exitCode,timeoutInfo = res
    if exitCode == 0:
        status = Test.PASS
    else:
        if timeoutInfo is None:
            status = Test.FAIL
        else:
            status = Test.TIMEOUT

    # Form the output log.
    output = """Script:\n--\n%s\n--\nExit Code: %d\n""" % (
        '\n'.join(script), exitCode)

    if timeoutInfo is not None:
        output += """Timeout: %s\n""" % (timeoutInfo,)
    output += "\n"

    # Append the outputs, if present.
    if out:
        output += """Command Output (stdout):\n--\n%s\n--\n""" % (out,)
    if err:
        output += """Command Output (stderr):\n--\n%s\n--\n""" % (err,)

    return lit.Test.Result(status, output)


def executeShTest(test, litConfig, useExternalSh,
                  extra_substitutions=[]):
    if test.config.unsupported:
        return lit.Test.Result(Test.UNSUPPORTED, 'Test is unsupported')

    script = parseIntegratedTestScript(test)
    if isinstance(script, lit.Test.Result):
        return script
    if litConfig.noExecute:
        return lit.Test.Result(Test.PASS)

    tmpDir, tmpBase = getTempPaths(test)
    substitutions = list(extra_substitutions)
    substitutions += getDefaultSubstitutions(test, tmpDir, tmpBase,
                                             normalize_slashes=useExternalSh)
    script = applySubstitutions(script, substitutions)

    # Re-run failed tests up to test_retry_attempts times.
    attempts = 1
    if hasattr(test.config, 'test_retry_attempts'):
        attempts += test.config.test_retry_attempts
    for i in range(attempts):
        res = _runShTest(test, litConfig, useExternalSh, script, tmpBase)
        if res.code != Test.FAIL:
            break
    # If we had to run the test more than once, count it as a flaky pass. These
    # will be printed separately in the test summary.
    if i > 0 and res.code == Test.PASS:
        res.code = Test.FLAKYPASS
    return res
