import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftTuple(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    @swiftTest
    @expectedFailureWindows
    def test(self):
        """Test the String formatter under adverse conditions"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        # FIXME: It would be even better if this were an error.
        zero = self.frame().FindVariable("zero")
        self.assertEqual(zero.GetSummary(), '<uninitialized>')
        random = self.frame().FindVariable("random")
        self.assertIn('cannot decode string', random.GetSummary())
        good = self.frame().FindVariable("good")
        self.assertIn('hello', good.GetSummary())
        options = good.GetTypeSummary().GetOptions()
        self.assertEqual(
            options & lldb.eTypeOptionHideChildren,
            lldb.eTypeOptionHideChildren,
            "String guts hidden",
        )
        self.assertEqual(
            good.GetSyntheticValue().GetNumChildren(), 0, "String guts hidden"
        )
