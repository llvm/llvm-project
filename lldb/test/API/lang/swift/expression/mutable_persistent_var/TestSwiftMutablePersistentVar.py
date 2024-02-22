import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftMutablePersistentVar(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    @swiftTest
    def test(self):
        """Test that persistent variables are mutable."""
        self.build()
        lldbutil.run_to_name_breakpoint(self, "main")
        self.expect("expr var $count = 30")
        self.expect("expr $count = 41")
        self.expect("expr $count += 1")
        self.expect("expr $count", substrs=["42"])
