import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftNSClassBaseClass(TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    @swiftTest
    @skipUnlessDarwin
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'))
        c = self.frame().FindVariable("c")
        c_type = c.GetType()
        self.assertIn("C", c_type.GetName())
        self.assertEqual(c_type.GetNumberOfDirectBaseClasses(), 1)
        nsobject = c_type.GetDirectBaseClassAtIndex(0)
        self.assertIn("NSObject", nsobject.GetName())
