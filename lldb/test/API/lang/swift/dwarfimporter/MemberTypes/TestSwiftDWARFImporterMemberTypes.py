import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftDWARFImporterC(lldbtest.TestBase):

    @swiftTest
    def test(self):
        """Test looking up a Clang typedef type that isn't directly
        referenced by debug info in the Swift object file.
        """
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        s = self.frame().FindVariable("s", lldb.eDynamicDontRunTarget)
        s_e = s.GetChildAtIndex(0)
        lldbutil.check_variable(self, s_e, value="someValue")

