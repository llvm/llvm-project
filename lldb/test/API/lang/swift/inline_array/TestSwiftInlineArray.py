import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftInlineArray(lldbtest.TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    
    @swiftTest
    def test(self):
        """Test the inline array synthetic child provider and summary"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        var_a = self.frame().FindVariable("a")
        self.assertEqual(var_a.GetSummary(), "4 values")
        synth_a = var_a.GetSyntheticValue()
        self.assertEqual(synth_a.GetNumChildren(), 4)
        self.assertEqual(synth_a.GetChildAtIndex(0).GetValue(), '4')
        self.assertEqual(synth_a.GetChildAtIndex(1).GetValue(), '3')
        self.assertEqual(synth_a.GetChildAtIndex(2).GetValue(), '2')
        self.assertEqual(synth_a.GetChildAtIndex(3).GetValue(), '1')
