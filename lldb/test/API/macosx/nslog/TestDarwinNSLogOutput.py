"""
Test DarwinLog "source include debug-level" functionality provided by the
StructuredDataDarwinLog plugin.

These tests are currently only supported when running against Darwin
targets.
"""


import lldb
import platform
import re
import sys

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbtest_config


class DarwinNSLogOutputTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @skipIfRemote   # this test is currently written using lldb commands & assumes running on local system

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.child = None
        self.child_prompt = '(lldb) '
        self.strict_sources = False

        # Source filename.
        self.source = 'main.m'

        # Output filename.
        self.exe_name = self.getBuildArtifact("a.out")
        self.d = {'OBJC_SOURCES': self.source, 'EXE': self.exe_name}

        # Locate breakpoint.
        self.line = line_number(self.source, '// break here')

    def tearDown(self):
        # Shut down the process if it's still running.
        if self.child:
            self.runCmd('process kill')
            self.expect_prompt()
            self.runCmd('quit')

        # Let parent clean up
        super(DarwinNSLogOutputTestCase, self).tearDown()

    def run_lldb_to_breakpoint(self, exe, source_file, line,
                               settings_commands=None):
        # Set self.child_prompt, which is "(lldb) ".
        prompt = self.child_prompt

        # So that the child gets torn down after the test.
        import pexpect
        self.child = pexpect.spawnu('%s %s %s' % (lldbtest_config.lldbExec,
                                                  self.lldbOption, exe))
        child = self.child

        # Turn on logging for what the child sends back.
        if self.TraceOn():
            child.logfile_read = sys.stdout

        # Disable showing of source lines at our breakpoint.
        # This is necessary for the logging tests, because the very
        # text we want to match for output from the running inferior
        # will show up in the source as well.  We don't want the source
        # output to erroneously make a match with our expected output.
        self.runCmd("settings set stop-line-count-before 0")
        self.expect_prompt()
        self.runCmd("settings set stop-line-count-after 0")
        self.expect_prompt()

        # Run any test-specific settings commands now.
        if settings_commands is not None:
            for setting_command in settings_commands:
                self.runCmd(setting_command)
                self.expect_prompt()

        # Set the breakpoint, and run to it.
        child.sendline('breakpoint set -f %s -l %d' % (source_file, line))
        child.expect_exact(prompt)
        child.sendline('run')
        child.expect_exact(prompt)

        # Ensure we stopped at a breakpoint.
        self.runCmd("thread list")
        self.expect(re.compile(r"stop reason = .*breakpoint"))

    def runCmd(self, cmd):
        if self.child:
            self.child.sendline(cmd)

    def expect_prompt(self, exactly=True):
        self.expect(self.child_prompt, exactly=exactly)

    def expect(self, pattern, exactly=False, *args, **kwargs):
        if exactly:
            return self.child.expect_exact(pattern, *args, **kwargs)
        return self.child.expect(pattern, *args, **kwargs)

    def do_test(self, expect_regexes=None, settings_commands=None):
        """ Run a test. """
        self.build(dictionary=self.d)
        self.setTearDownCleanup(dictionary=self.d)

        exe = self.getBuildArtifact(self.exe_name)
        self.run_lldb_to_breakpoint(exe, self.source, self.line,
                                    settings_commands=settings_commands)
        self.expect_prompt()

        # Now go.
        self.runCmd("process continue")
        self.expect(expect_regexes)

    def test_nslog_output_is_displayed(self):
        """Test that NSLog() output shows up in the command-line debugger."""
        self.do_test(expect_regexes=[
            re.compile(r"(This is a message from NSLog)"),
            re.compile(r"Process \d+ exited with status")
        ])
        self.assertIsNotNone(self.child.match)
        self.assertGreater(len(self.child.match.groups()), 0)
        self.assertEqual(
            "This is a message from NSLog",
            self.child.match.group(1))

    def test_nslog_output_is_suppressed_with_env_var(self):
        """Test that NSLog() output does not show up with the ignore env var."""
        # This test will only work properly on macOS 10.12+.  Skip it on earlier versions.
        # This will require some tweaking on iOS.
        match = re.match(r"^\d+\.(\d+)", platform.mac_ver()[0])
        if match is None or int(match.group(1)) < 12:
            self.skipTest("requires macOS 10.12 or higher")

        self.do_test(
            expect_regexes=[
                re.compile(r"(This is a message from NSLog)"),
                re.compile(r"Process \d+ exited with status")
            ],
            settings_commands=[
                "settings set target.env-vars "
                "\"IDE_DISABLED_OS_ACTIVITY_DT_MODE=1\""
            ])
        self.assertIsNotNone(self.child.match)
        self.assertEqual(len(self.child.match.groups()), 0)
