import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))
        self.runCmd("type summary add -s '${var.ubyte%%2.2X}${var.sbyte%%2.2X}!' Bytes")
        self.expect("v bytes", substrs=[" = 1001!"])
