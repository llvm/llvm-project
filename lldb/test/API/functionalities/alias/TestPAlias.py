import lldb
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "return", lldb.SBFileSpec("main.c"))
        self.expect("p -g", substrs=["$0 = -"])
        self.expect("p -i0 -g", error=True)
