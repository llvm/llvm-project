"""Test lldb's response time for 'frame variable' command."""

from __future__ import print_function


import sys
import lldb
from lldbsuite.test import configuration
from lldbsuite.test import lldbtest_config
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbbench import *


class FrameVariableResponseBench(BenchBase):

    def setUp(self):
        BenchBase.setUp(self)
        self.exe = lldbtest_config.lldbExec
        self.break_spec = '-n main'
        self.count = 20

    @benchmarks_test
    @no_debug_info_test
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr22274: need a pexpect replacement for windows")
    def test_startup_delay(self):
        """Test response time for the 'frame variable' command."""
        print()
        self.run_frame_variable_bench(self.exe, self.break_spec, self.count)
        print("lldb frame variable benchmark:", self.stopwatch)

    def run_frame_variable_bench(self, exe, break_spec, count):
        import pexpect
        # Set self.child_prompt, which is "(lldb) ".
        self.child_prompt = '(lldb) '
        prompt = self.child_prompt

        # Reset the stopwatchs now.
        self.stopwatch.reset()
        for i in range(count):
            # So that the child gets torn down after the test.
            self.child = pexpect.spawn(
                '%s %s %s' %
                (lldbtest_config.lldbExec, self.lldbOption, exe))
            child = self.child

            # Turn on logging for what the child sends back.
            if self.TraceOn():
                child.logfile_read = sys.stdout

            # Set our breakpoint.
            child.sendline('breakpoint set %s' % break_spec)
            child.expect_exact(prompt)

            # Run the target and expect it to be stopped due to breakpoint.
            child.sendline('run')  # Aka 'process launch'.
            child.expect_exact(prompt)

            with self.stopwatch:
                # Measure the 'frame variable' response time.
                child.sendline('frame variable')
                child.expect_exact(prompt)

            child.sendline('quit')
            try:
                self.child.expect(pexpect.EOF)
            except:
                pass

        # The test is about to end and if we come to here, the child process has
        # been terminated.  Mark it so.
        self.child = None
