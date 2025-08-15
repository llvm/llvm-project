import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftConstrainedExistential(lldbtest.TestBase):
    @swiftTest
    def test(self):
        """Test constrained existential types"""

        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec("main.swift"))
        s0 = self.frame().FindVariable("s0")
        self.assertEqual(s0.GetStaticValue().GetNumChildren(), 3+1+2)
        self.assertEqual(s0.GetNumChildren(), 1)
        i = s0.GetChildMemberWithName("i")
        lldbutil.check_variable(self, i, value='23')

        s = self.frame().FindVariable("s")
        s = s.GetChildAtIndex(0)
        self.assertEqual(s.GetStaticValue().GetNumChildren(), 6)
        self.assertEqual(s.GetNumChildren(), 1)
        i = s.GetChildMemberWithName("i")
        lldbutil.check_variable(self, i, value='23')
