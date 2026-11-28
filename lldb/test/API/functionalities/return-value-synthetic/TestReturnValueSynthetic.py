"""
Test that the return value in the thread description uses synthetic children.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@requireNotWasm("return value is unrecoverable from the operand stack at a step-out")
class TestCase(TestBase):
    @skipIf(archs=["arm$"], oslist=["freebsd", "linux"], bugnumber="llvm.org/pr44132")
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))
        self.runCmd(
            "command script import " + os.path.join(self.getSourceDir(), "synthetic.py")
        )
        self.runCmd("type synthetic add -l synthetic.PairProvider Pair")
        self.runCmd("finish")
        self.expect(
            "thread info",
            patterns=[r"Return value: \(Pair\) \$\d+ = \(second = 2\)\n"],
        )
