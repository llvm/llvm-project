"""Test lldb's expression evaluations and collect statistics."""

import sys
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbbench import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import configuration
from lldbsuite.test import lldbutil


class ExpressionEvaluationCase(BenchBase):
    def setUp(self):
        BenchBase.setUp(self)
        self.source = "main.cpp"
        self.line_to_break = line_number(self.source, "// Set breakpoint here.")
        self.count = 25

    @benchmarks_test
    @add_test_categories(["pexpect"])
    def test_expr_cmd(self):
        """Test lldb's expression commands and collect statistics."""
        self.build()
        self.exe_name = "a.out"

        print()
        self.run_lldb_repeated_exprs(self.exe_name, self.count)
        print("lldb expr cmd benchmark:", self.stopwatch)

    def run_lldb_repeated_exprs(self, exe_name, count):
        import pexpect

        exe = self.getBuildArtifact(exe_name)

        # Set self.child_prompt, which is "(lldb) ".
        self.child_prompt = "(lldb) "
        prompt = self.child_prompt

        # Reset the stopwatch now.
        self.stopwatch.reset()
        for i in range(count):
            # So that the child gets torn down after the test.
            self.child = pexpect.spawn(
                "%s %s %s" % (lldbtest_config.lldbExec, self.lldbOption, exe)
            )
            child = self.child

            # Turn on logging for what the child sends back.
            if self.TraceOn():
                child.logfile_read = sys.stdout

            child.expect_exact(prompt)
            child.sendline(
                "breakpoint set -f %s -l %d" % (self.source, self.line_to_break)
            )
            child.expect_exact(prompt)
            child.sendline("run")
            child.expect_exact(prompt)
            expr_cmd1 = "expr ptr[j]->point.x"
            expr_cmd2 = "expr ptr[j]->point.y"

            with self.stopwatch:
                child.sendline(expr_cmd1)
                child.expect_exact(prompt)
                child.sendline(expr_cmd2)
                child.expect_exact(prompt)

            child.sendline("quit")
            try:
                self.child.expect(pexpect.EOF)
            except:
                pass

        self.child = None
