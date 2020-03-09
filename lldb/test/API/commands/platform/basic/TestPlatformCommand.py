"""
Test some lldb platform commands.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class PlatformCommandTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_help_platform(self):
        self.runCmd("help platform")

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
                'Kernel',
                'Hostname'])

    @expectedFailureAll(oslist=["windows"])
    @no_debug_info_test
    def test_shell(self):
        """ Test that the platform shell command can invoke ls. """
        triple = self.dbg.GetSelectedPlatform().GetTriple()
        if re.match(".*-.*-windows", triple):
            self.expect(
                "platform shell dir c:\\", substrs=[
                    "Windows", "Program Files"])
        elif re.match(".*-.*-.*-android", triple):
            self.expect(
                "platform shell ls /",
                substrs=[
                    "cache",
                    "dev",
                    "system"])
        else:
            self.expect("platform shell ls /", substrs=["dev", "tmp", "usr"])

    @no_debug_info_test
    def test_shell_builtin(self):
        """ Test a shell built-in command (echo) """
        self.expect("platform shell echo hello lldb",
                    substrs=["hello lldb"])

    # FIXME: re-enable once platform shell -t can specify the desired timeout
    @no_debug_info_test
    def test_shell_timeout(self):
        """ Test a shell built-in command (sleep) that times out """
        self.skipTest("due to taking too long to complete.")
        self.expect("platform shell sleep 15", error=True, substrs=[
                    "error: timed out waiting for shell command to complete"])
