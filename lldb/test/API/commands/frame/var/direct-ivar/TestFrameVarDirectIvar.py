import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test_cpp_this(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "this", lldb.SBFileSpec("main.mm"))
        self.expect("frame variable m_field", startstr="(int) m_field = 30")

    def test_objc_self(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "self", lldb.SBFileSpec("main.mm"))
        self.expect("frame variable _ivar", startstr="(int) _ivar = 41")
