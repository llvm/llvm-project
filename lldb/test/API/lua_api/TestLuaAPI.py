"""
Test Lua API wrapper
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import subprocess

def to_string(b):
    """Return the parameter as type 'str', possibly encoding it.

    In Python2, the 'str' type is the same as 'bytes'. In Python3, the
    'str' type is (essentially) Python2's 'unicode' type, and 'bytes' is
    distinct.

    """
    if isinstance(b, str):
        # In Python2, this branch is taken for types 'str' and 'bytes'.
        # In Python3, this branch is taken only for 'str'.
        return b
    if isinstance(b, bytes):
        # In Python2, this branch is never taken ('bytes' is handled as 'str').
        # In Python3, this is true only for 'bytes'.
        try:
            return b.decode('utf-8')
        except UnicodeDecodeError:
            # If the value is not valid Unicode, return the default
            # repr-line encoding.
            return str(b)

    # By this point, here's what we *don't* have:
    #
    #  - In Python2:
    #    - 'str' or 'bytes' (1st branch above)
    #  - In Python3:
    #    - 'str' (1st branch above)
    #    - 'bytes' (2nd branch above)
    #
    # The last type we might expect is the Python2 'unicode' type. There is no
    # 'unicode' type in Python3 (all the Python3 cases were already handled). In
    # order to get a 'str' object, we need to encode the 'unicode' object.
    try:
        return b.encode('utf-8')
    except AttributeError:
        raise TypeError('not sure how to convert %s to %s' % (type(b), str))

class ExecuteCommandTimeoutException(Exception):
    def __init__(self, msg, out, err, exitCode):
        assert isinstance(msg, str)
        assert isinstance(out, str)
        assert isinstance(err, str)
        assert isinstance(exitCode, int)
        self.msg = msg
        self.out = out
        self.err = err
        self.exitCode = exitCode


# Close extra file handles on UNIX (on Windows this cannot be done while
# also redirecting input).
kUseCloseFDs = not (platform.system() == 'Windows')


def executeCommand(command, cwd=None, env=None, input=None, timeout=0):
    """Execute command ``command`` (list of arguments or string) with.

    * working directory ``cwd`` (str), use None to use the current
    working directory
    * environment ``env`` (dict), use None for none
    * Input to the command ``input`` (str), use string to pass
    no input.
    * Max execution time ``timeout`` (int) seconds. Use 0 for no timeout.

    Returns a tuple (out, err, exitCode) where
    * ``out`` (str) is the standard output of running the command
    * ``err`` (str) is the standard error of running the command
    * ``exitCode`` (int) is the exitCode of running the command

    If the timeout is hit an ``ExecuteCommandTimeoutException``
    is raised.

    """
    if input is not None:
        input = to_bytes(input)
    p = subprocess.Popen(command, cwd=cwd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=env, close_fds=kUseCloseFDs)
    timerObject = None
    # FIXME: Because of the way nested function scopes work in Python 2.x we
    # need to use a reference to a mutable object rather than a plain
    # bool. In Python 3 we could use the "nonlocal" keyword but we need
    # to support Python 2 as well.
    hitTimeOut = [False]
    try:
        if timeout > 0:
            def killProcess():
                # We may be invoking a shell so we need to kill the
                # process and all its children.
                hitTimeOut[0] = True
                killProcessAndChildren(p.pid)

            timerObject = threading.Timer(timeout, killProcess)
            timerObject.start()

        out, err = p.communicate(input=input)
        exitCode = p.wait()
    finally:
        if timerObject != None:
            timerObject.cancel()

    # Ensure the resulting output is always of string type.
    out = to_string(out)
    err = to_string(err)

    if hitTimeOut[0]:
        raise ExecuteCommandTimeoutException(
            msg='Reached timeout of {} seconds'.format(timeout),
            out=out,
            err=err,
            exitCode=exitCode
        )

    # Detect Ctrl-C in subprocess.
    if exitCode == -signal.SIGINT:
        raise KeyboardInterrupt

    return out, err, exitCode

class TestLuaAPI(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def get_tests(self):
        tests = []
        for filename in os.listdir():
            # Ignore dot files and excluded tests.
            if filename.startswith('.'):
                continue

            # Ignore files that don't start with 'Test'.
            if not filename.startswith('Test'):
                continue

            if not os.path.isdir(filename):
                base, ext = os.path.splitext(filename)
                if ext == '.lua':
                    tests.append(filename)
        return tests

    def test_lua_api(self):  
        if "LUA_EXECUTABLE" not in os.environ or len(os.environ["LUA_EXECUTABLE"]) == 0:
           self.skipTest("Lua API tests could not find Lua executable.")
           return
        lua_executable = os.environ["LUA_EXECUTABLE"]

        self.build()
        test_exe = self.getBuildArtifact("a.out")
        test_output = self.getBuildArtifact("output")
        test_input = self.getBuildArtifact("input")

        lua_lldb_cpath = "%s/lua/5.3/?.so" % configuration.lldb_libs_dir

        lua_prelude = "package.cpath = '%s;' .. package.cpath" % lua_lldb_cpath

        lua_env = {
            "TEST_EXE":     os.path.join(self.getBuildDir(), test_exe),
            "TEST_OUTPUT":  os.path.join(self.getBuildDir(), test_output),
            "TEST_INPUT":   os.path.join(self.getBuildDir(), test_input)
        }

        for lua_test in self.get_tests():
            cmd = [lua_executable] + ["-e", lua_prelude] + [lua_test]
            out, err, exitCode = executeCommand(cmd, env=lua_env)

            # Redirect Lua output
            print(out)
            print(err, file=sys.stderr)

            self.assertTrue(
                exitCode == 0,
                "Lua test '%s' failure." % lua_test
            )
