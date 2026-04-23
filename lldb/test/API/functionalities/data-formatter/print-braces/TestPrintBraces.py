import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPrintBraces(TestBase):
    def test_default_has_braces(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))
        self.expect("frame variable s", substrs=["{", "}"])

    def test_no_braces(self):
        self.build()
        self.runCmd("settings set target.print-braces false")
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))
        self.expect("frame variable s", matching=False, substrs=["{", "}"])
        self.expect("frame variable s", substrs=["x = 1", "y = 2", "z = 3"])
