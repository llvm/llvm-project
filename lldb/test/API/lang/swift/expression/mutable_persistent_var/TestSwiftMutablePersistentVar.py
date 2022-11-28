import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @swiftTest
    def test(self):
        """Test that persistent variables are mutable."""
        self.build()
        lldbutil.run_to_name_breakpoint(self, "main")
        self.expect("expr var $count = 30")
        self.expect("expr $count = 41")
