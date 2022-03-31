"""
Test some lldb platform commands.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class PlatformCommandTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @no_debug_info_test
    def test_help_platform(self):
        self.runCmd("help platform")

    @no_debug_info_test
    def test_help_shell_alias(self):
        self.expect("help shell", substrs=["Run a shell command on the host.",
                                           "shell <shell-command>",
                                           "'shell' is an abbreviation"])
        # "platform shell" has options. The "shell" alias for it does not.
        self.expect("help shell", substrs=["Command Options:"], matching=False)

    @no_debug_info_test
    def test_list(self):
        self.expect("platform list",
                    patterns=['^Available platforms:'])

    @no_debug_info_test
    def test_process_list(self):
        self.expect("platform process list",
                    substrs=['PID', 'TRIPLE', 'NAME'])

    @no_debug_info_test
    def test_process_info_with_no_arg(self):
        """This is expected to fail and to return a proper error message."""
        self.expect("platform process info", error=True,
                    substrs=['one or more process id(s) must be specified'])

    @no_debug_info_test
    def test_status(self):
        self.expect(
            "platform status",
            substrs=[
                'Platform',
                'Triple',
                'OS Version',
                'Hostname',
                'Kernel',
            ])

    @expectedFailureAll(oslist=["windows"])
    @no_debug_info_test
    def test_shell(self):
        """ Test that the platform shell command can invoke ls. """
        triple = self.dbg.GetSelectedPlatform().GetTriple()
        if re.match(".*-.*-windows", triple):
            self.expect(
                "platform shell dir c:\\", substrs=[
                    "Windows", "Program Files"])
            self.expect("shell dir c:\\", substrs=["Windows", "Program Files"])
        elif re.match(".*-.*-.*-android", triple):
            self.expect(
                "platform shell ls /",
                substrs=[
                    "cache",
                    "dev",
                    "system"])
            self.expect("shell ls /",
                substrs=["cache", "dev", "system"])
        else:
            self.expect("platform shell ls /", substrs=["dev", "tmp", "usr"])
            self.expect("shell ls /", substrs=["dev", "tmp", "usr"])

    @no_debug_info_test
    def test_shell_builtin(self):
        """ Test a shell built-in command (echo) """
        self.expect("platform shell echo hello lldb",
                    substrs=["hello lldb"])
        self.expect("shell echo hello lldb",
                    substrs=["hello lldb"])


    @no_debug_info_test
    def test_shell_timeout(self):
        """ Test a shell built-in command (sleep) that times out """
        self.skipTest("Alias with option not supported by the command interpreter.")
        self.expect("platform shell -t 1 -- sleep 15", error=True, substrs=[
                    "error: timed out waiting for shell command to complete"])
        self.expect("shell -t 1 --  sleep 3", error=True, substrs=[
                    "error: timed out waiting for shell command to complete"])

    @no_debug_info_test
    def test_host_shell_interpreter(self):
        """ Test the host platform shell with a different interpreter """
        self.build()
        exe = self.getBuildArtifact('a.out')
        self.expect("platform shell -h -s " + exe + " -- 'echo $0'",
                    substrs=['SUCCESS', 'a.out'])
