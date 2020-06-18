"""
Test children counts of pointer values.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPointerNumChilden(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    def test_pointer_num_children(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))

        result = self.frame().FindVariable("Ref")
        self.assertEqual(1, result.GetNumChildren())
        self.assertEqual(2, result.GetChildAtIndex(0).GetNumChildren())
        self.assertEqual("42", result.GetChildAtIndex(0).GetChildAtIndex(0).GetValue())
        self.assertEqual("56", result.GetChildAtIndex(0).GetChildAtIndex(1).GetValue())

        result = self.frame().FindVariable("Ptr")
        self.assertEqual(1, result.GetNumChildren())
        self.assertEqual(2, result.GetChildAtIndex(0).GetNumChildren())
        self.assertEqual("42", result.GetChildAtIndex(0).GetChildAtIndex(0).GetValue())
        self.assertEqual("56", result.GetChildAtIndex(0).GetChildAtIndex(1).GetValue())
