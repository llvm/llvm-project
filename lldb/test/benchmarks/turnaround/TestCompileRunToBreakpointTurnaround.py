"""Benchmark the turnaround time starting a debugger and run to the breakpont with lldb vs. gdb."""

import os, sys
import unittest2
import lldb
import pexpect
from lldbbench import *

class CompileRunToBreakpointBench(BenchBase):

    mydir = os.path.join("benchmarks", "turnaround")

    def setUp(self):
        BenchBase.setUp(self)
        self.exe = self.lldbHere
        self.function = 'Driver::MainLoop()'

        self.count = lldb.bmIterationCount
        if self.count <= 0:
            self.count = 3

        self.lldb_avg = None
        self.gdb_avg = None

    @benchmarks_test
    def test_run_lldb_then_gdb(self):
        """Benchmark turnaround time with lldb vs. gdb."""
        print
        self.run_lldb_turnaround(self.exe, self.function, self.count)
        print "lldb turnaround benchmark:", self.stopwatch
        self.run_gdb_turnaround(self.exe, self.function, self.count)
        print "gdb turnaround benchmark:", self.stopwatch
        print "lldb_avg/gdb_avg: %f" % (self.lldb_avg/self.gdb_avg)

    def run_lldb_turnaround(self, exe, function, count):
        def run_one_round():
            prompt = self.child_prompt

            # So that the child gets torn down after the test.
            self.child = pexpect.spawn('%s %s %s' % (self.lldbExec, self.lldbOption, exe))
            child = self.child

            # Turn on logging for what the child sends back.
            if self.TraceOn():
                child.logfile_read = sys.stdout

            child.expect_exact(prompt)
            child.sendline('breakpoint set -F %s' % function)
            child.expect_exact(prompt)
            child.sendline('run')
            child.expect_exact(prompt)

        # Set self.child_prompt, which is "(lldb) ".
        self.child_prompt = '(lldb) '
        # Reset the stopwatch now.
        self.stopwatch.reset()

        for i in range(count + 1):
            # Ignore the first invoke lldb and run to the breakpoint turnaround time.
            if i == 0:
                run_one_round()
            else:
                with self.stopwatch:
                    run_one_round()

            self.child.sendline('quit')
            try:
                self.child.expect(pexpect.EOF)
            except:
                pass

        self.lldb_avg = self.stopwatch.avg()
        self.child = None

    def run_gdb_turnaround(self, exe, function, count):
        def run_one_round():
            prompt = self.child_prompt

            # So that the child gets torn down after the test.
            self.child = pexpect.spawn('gdb --nx %s' % exe)
            child = self.child

            # Turn on logging for what the child sends back.
            if self.TraceOn():
                child.logfile_read = sys.stdout

            child.expect_exact(prompt)
            child.sendline('break %s' % function)
            child.expect_exact(prompt)
            child.sendline('run')
            child.expect_exact(prompt)

        # Set self.child_prompt, which is "(gdb) ".
        self.child_prompt = '(gdb) '
        # Reset the stopwatch now.
        self.stopwatch.reset()

        for i in range(count+1):
            # Ignore the first invoke lldb and run to the breakpoint turnaround time.
            if i == 0:
                run_one_round()
            else:
                with self.stopwatch:
                    run_one_round()

            self.child.sendline('quit')
            self.child.expect_exact('The program is running.  Exit anyway?')
            self.child.sendline('y')
            try:
                self.child.expect(pexpect.EOF)
            except:
                pass

        self.gdb_avg = self.stopwatch.avg()
        self.child = None


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
