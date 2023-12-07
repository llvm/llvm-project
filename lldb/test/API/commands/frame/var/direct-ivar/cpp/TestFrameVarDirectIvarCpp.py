import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test_cpp_this(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// check this", lldb.SBFileSpec("main.cpp")
        )
        self.expect("frame variable m_field", startstr="(int) m_field = 30")
