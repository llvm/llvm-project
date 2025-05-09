import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        """Verify summary formatter for tagged strings."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.m"))
        self.expect("v str1 str2", patterns=['@"nineDigit"', '@"tenDigitXX"'])
