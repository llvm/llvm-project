"""
Test @lldb.summary and @lldb.synthetic decorators lead to automatic formatter
registration, when using `command script import`.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_summary(self):
        self.build()
        self.runCmd("command script import formatters.py")
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )
        frame = thread.selected_frame
        p = frame.var("p")
        self.assertEqual(p.summary, "(1, 2)")

    def test_synthetic(self):
        self.build()
        self.runCmd("command script import formatters.py")
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )
        self.expect("v c", substrs=["[0] = 10", "[1] = 20", "[2] = 30"])
