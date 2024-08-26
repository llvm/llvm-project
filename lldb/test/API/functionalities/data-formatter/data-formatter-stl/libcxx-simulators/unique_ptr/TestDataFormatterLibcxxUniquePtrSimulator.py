"""
Test we can understand various layouts of the libc++'s std::unique_ptr
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxUniquePtrDataFormatterSimulatorTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("main.cpp")
        )
        self.expect("frame variable var_up", substrs=["pointer ="])
        self.expect("frame variable var_up", substrs=["deleter ="], matching=False)
        self.expect(
            "frame variable var_with_deleter_up", substrs=["pointer =", "deleter ="]
        )
