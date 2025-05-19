import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftClassBaseClass(TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    @swiftTest
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'))
        c = self.frame().FindVariable("c")
        c1 = c.GetType()
        self.assertIn("C1", c1.GetName())
        self.assertEqual(c1.GetNumberOfDirectBaseClasses(), 1)
        c0 = c1.GetDirectBaseClassAtIndex(0)
        self.assertIn("C0", c0.GetName())
