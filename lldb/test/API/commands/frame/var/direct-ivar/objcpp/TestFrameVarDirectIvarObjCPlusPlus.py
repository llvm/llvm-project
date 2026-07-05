import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @skipUnlessDarwin
    def test_objc_self(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "check self", lldb.SBFileSpec("main.mm")
        )
        self.expect("frame variable _ivar", startstr="(int) _ivar = 30")

    @skipUnlessDarwin
    def test_objc_explicit_self(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "check explicit self", lldb.SBFileSpec("main.mm")
        )
        self.expect("frame variable _ivar", startstr="(int) _ivar = 30")

    @skipUnlessDarwin
    def test_cpp_this(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "check this", lldb.SBFileSpec("main.mm")
        )
        self.expect("frame variable m_field", startstr="(int) m_field = 41")
