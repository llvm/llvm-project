import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSBValueSynthetic(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_str(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        vector = self.frame().FindVariable("vector")
        has_vector = self.frame().FindVariable("has_vector")
        self.expect(str(vector), exe=False, substrs=["42", "47"])
        self.expect(str(has_vector), exe=False, substrs=["42", "47"])
