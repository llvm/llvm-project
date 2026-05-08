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
        ic = frame.var("ic")
        self.assertEqual(ic.summary, "size=2")
        fc = frame.var("fc")
        self.assertEqual(fc.summary, "size=2")

    def test_synthetic(self):
        self.build()
        self.runCmd("command script import formatters.py")
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )
        self.expect("v ic", substrs=["[0] = 10", "[1] = 20"])
        self.expect("v fc", substrs=["[0] = 10.5", "[1] = 20.25"])

    def test_failure(self):
        self.expect("command script import broken_formatter.py", error=True)
