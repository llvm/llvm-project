"""Test lldb's response time for 'frame variable' command."""

import os, sys
import unittest2
import lldb
import pexpect
from lldbbench import *

class FrameVariableResponseBench(BenchBase):

    mydir = os.path.join("benchmarks", "frame_variable")

    def setUp(self):
        BenchBase.setUp(self)
        if lldb.bmExecutable:
            self.exe = lldb.bmExecutable
        else:
            self.exe = self.lldbHere
        if lldb.bmBreakpointSpec:
            self.break_spec = lldb.bmBreakpointSpec
        else:
            self.break_spec = '-n main'

        self.count = lldb.bmIterationCount
        if self.count <= 0:
            self.count = 20

    @benchmarks_test
    def test_startup_delay(self):
        """Test response time for the 'frame variable' command."""
        print
        self.run_frame_variable_bench(self.exe, self.break_spec, self.count)
        print "lldb frame variable benchmark:", self.stopwatch

    def run_frame_variable_bench(self, exe, break_spec, count):
        # Set self.child_prompt, which is "(lldb) ".
        self.child_prompt = '(lldb) '
        prompt = self.child_prompt

        # Reset the stopwatchs now.
        self.stopwatch.reset()
        for i in range(count):
            # So that the child gets torn down after the test.
            self.child = pexpect.spawn('%s %s %s' % (self.lldbHere, self.lldbOption, exe))
            child = self.child

            # Turn on logging for what the child sends back.
            if self.TraceOn():
                child.logfile_read = sys.stdout

            # Set our breakpoint.
            child.sendline('breakpoint set %s' % break_spec)
            child.expect_exact(prompt)

            # Run the target and expect it to be stopped due to breakpoint.
            child.sendline('run') # Aka 'process launch'.
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


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
