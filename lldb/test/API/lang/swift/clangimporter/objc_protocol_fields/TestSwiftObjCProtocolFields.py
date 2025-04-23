import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftObjCBaseClassMemberLookup(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Test accessing a static member from a member function"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift')
        )

        p = self.frame().FindVariable("p").GetStaticValue()
        self.assertEqual(p.GetNumChildren(), 1)
        self.assertEqual(p.GetChildAtIndex(0).GetSummary(), '"hello"')
