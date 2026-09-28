import lldb
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    TEST_WITH_PDB_DEBUG_INFO = True

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "return", lldb.SBFileSpec("main.c"))
        self.expect("p -g", startstr="(int) -41")
        self.expect("p -i0 -g", error=True)
