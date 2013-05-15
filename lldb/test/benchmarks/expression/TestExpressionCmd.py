"""Test lldb's expression evaluations and collect statistics."""

import os, sys
import unittest2
import lldb
import pexpect
from lldbbench import *

class ExpressionEvaluationCase(BenchBase):

    mydir = os.path.join("benchmarks", "expression")

    def setUp(self):
        BenchBase.setUp(self)
        self.source = 'main.cpp'
        self.line_to_break = line_number(self.source, '// Set breakpoint here.')
        self.count = lldb.bmIterationCount
        if self.count <= 0:
            self.count = 25

    @benchmarks_test
    def test_expr_cmd(self):
        """Test lldb's expression commands and collect statistics."""
        self.buildDefault()
        self.exe_name = 'a.out'

        print
        self.run_lldb_repeated_exprs(self.exe_name, self.count)
        print "lldb expr cmd benchmark:", self.stopwatch

    def run_lldb_repeated_exprs(self, exe_name, count):
        exe = os.path.join(os.getcwd(), exe_name)

        # Set self.child_prompt, which is "(lldb) ".
        self.child_prompt = '(lldb) '
        prompt = self.child_prompt

        # Reset the stopwatch now.
        self.stopwatch.reset()
        for i in range(count):
            # So that the child gets torn down after the test.
            self.child = pexpect.spawn('%s %s %s' % (self.lldbHere, self.lldbOption, exe))
            child = self.child

            # Turn on logging for what the child sends back.
            if self.TraceOn():
                child.logfile_read = sys.stdout

            child.expect_exact(prompt)
            child.sendline('breakpoint set -f %s -l %d' % (self.source, self.line_to_break))
            child.expect_exact(prompt)
            child.sendline('run')
            child.expect_exact(prompt)
            expr_cmd1 = 'expr ptr[j]->point.x'
            expr_cmd2 = 'expr ptr[j]->point.y'

            with self.stopwatch:
                child.sendline(expr_cmd1)
                child.expect_exact(prompt)
                child.sendline(expr_cmd2)
                child.expect_exact(prompt)

            child.sendline('quit')
            try:
                self.child.expect(pexpect.EOF)
            except:
                pass

        self.child = None


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
