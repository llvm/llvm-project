"""
Check that SBValue.GetValueAsSigned() does the right thing for a 32-bit -1.
"""

import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_with_run_command(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self,"// break here", lldb.SBFileSpec("main.cpp"))

        self.assertEqual(self.frame().FindVariable("myvar").GetValueAsSigned(), -1)
        self.assertEqual(self.frame().FindVariable("myvar").GetValueAsUnsigned(), 0xFFFFFFFF)
