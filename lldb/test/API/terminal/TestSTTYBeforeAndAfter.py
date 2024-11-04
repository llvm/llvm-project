"""
Test that 'stty -a' displays the same output before and after running the lldb command.
"""

import lldb
import io
import sys
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSTTYBeforeAndAfter(TestBase):
    @classmethod
    def classCleanup(cls):
        """Cleanup the test byproducts."""
        cls.RemoveTempFile("child_send1.txt")
        cls.RemoveTempFile("child_read1.txt")
        cls.RemoveTempFile("child_send2.txt")
        cls.RemoveTempFile("child_read2.txt")

    @expectedFailureAll(
        hostoslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows",
    )
    @no_debug_info_test
    def test_stty_dash_a_before_and_afetr_invoking_lldb_command(self):
        """Test that 'stty -a' displays the same output before and after running the lldb command."""
        import pexpect

        if not which("expect"):
            self.skipTest("The 'expect' program cannot be located, skip the test")

        # The expect prompt.
        expect_prompt = "expect[0-9.]+> "
        # The default lldb prompt.
        lldb_prompt = "(lldb) "

        # So that the child gets torn down after the test.
        self.child = pexpect.spawn("expect", encoding="utf-8")
        child = self.child

        child.expect(expect_prompt)
        child.setecho(True)
        if self.TraceOn():
            child.logfile = sys.stdout

        if self.platformIsDarwin():
            child.sendline("set env(TERM) xterm")
        else:
            child.sendline("set env(TERM) vt100")
        child.expect(expect_prompt)
        child.sendline("puts $env(TERM)")
        child.expect(expect_prompt)

        # Turn on loggings for input/output to/from the child.
        child.logfile_send = child_send1 = io.StringIO()
        child.logfile_read = child_read1 = io.StringIO()
        child.sendline("stty -a")
        child.expect(expect_prompt)

        # Now that the stage1 logging is done, restore logfile to None to
        # stop further logging.
        child.logfile_send = None
        child.logfile_read = None

        # Invoke the lldb command.
        child.sendline(lldbtest_config.lldbExec)
        child.expect_exact(lldb_prompt)

        # Immediately quit.
        child.sendline("quit")
        child.expect(expect_prompt)

        child.logfile_send = child_send2 = io.StringIO()
        child.logfile_read = child_read2 = io.StringIO()
        child.sendline("stty -a")
        child.expect(expect_prompt)

        child.sendline("exit")

        # Now that the stage2 logging is done, restore logfile to None to
        # stop further logging.
        child.logfile_send = None
        child.logfile_read = None

        if self.TraceOn():
            print("\n\nContents of child_send1:")
            print(child_send1.getvalue())
            print("\n\nContents of child_read1:")
            print(child_read1.getvalue())
            print("\n\nContents of child_send2:")
            print(child_send2.getvalue())
            print("\n\nContents of child_read2:")
            print(child_read2.getvalue())

        stty_output1_lines = child_read1.getvalue().splitlines()
        stty_output2_lines = child_read2.getvalue().splitlines()
        zipped = list(zip(stty_output1_lines, stty_output2_lines))
        for tuple in zipped:
            if self.TraceOn():
                print("tuple->%s" % str(tuple))
            # Every line should compare equal until the first blank line.
            if len(tuple[0]) == 0:
                break
            self.assertEqual(tuple[0], tuple[1])
