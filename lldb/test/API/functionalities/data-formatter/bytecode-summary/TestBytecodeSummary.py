import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @skipUnlessDarwin
    def test(self):
        self.build()
        if self.TraceOn():
            self.expect("log enable -v lldb formatters")
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )
        self.expect("v x", substrs=["(MyOptional<int>) x = None"])
        self.expect("v y", substrs=["(MyOptional<int>) y = 42"])
