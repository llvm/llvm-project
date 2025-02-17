"""
Test dwim-print with objc instances.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @skipUnlessDarwin
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.m"))
        self.expect("dwim-print obj", substrs=["_number = 15"])
