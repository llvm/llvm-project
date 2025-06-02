import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    @swiftTest
    def test(self):
        """Test indirect enums"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect("v")
        self.expect("frame variable s", substrs = ["EmptyBase.S", "16"])
        self.expect("target variable svar", substrs = ["32"])
        self.expect("frame variable e", substrs = ["baseCase"])
