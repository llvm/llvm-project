import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class ArrayTypedefTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_array_typedef(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here",
            lldb.SBFileSpec("main.cpp", False))
        self.expect("expr str", substrs=['"abcd"'])
