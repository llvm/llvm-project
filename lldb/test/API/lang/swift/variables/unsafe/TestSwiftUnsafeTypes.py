import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftUnsafeTypes(TestBase):

    def check_ptr(self):
        ptr = self.frame().FindVariable("ptr")
        self.assertEqual(ptr.GetSummary()[:2], '0x')
        self.assertEqual(ptr.GetNumChildren(), 1)
        child = ptr.GetChildAtIndex(0)
        self.assertEqual(child.GetName(), 'pointee')
        self.assertEqual(child.GetSummary(), 'nil')

    @swiftTest
    def test(self):
        """Test formatters for unsafe types"""
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(self, 'break here',
                                          lldb.SBFileSpec('main.swift'))
        self.check_ptr()
        process.Continue()
        self.check_ptr()
