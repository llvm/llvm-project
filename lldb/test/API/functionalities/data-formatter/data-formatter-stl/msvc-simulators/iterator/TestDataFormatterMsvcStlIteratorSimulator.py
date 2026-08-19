"""Simulate MSVC STL vector iterators and check the formatter."""

from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MsvcStlIteratorSimulatorTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        it = self.frame().FindVariable("it")
        self.assertEqual(it.GetNumChildren(), 1)
        self.assertEqual(it.GetChildAtIndex(0).GetName(), "item")
        self.assertEqual(it.GetChildAtIndex(0).GetValueAsSigned(), 3)
        self.expect("frame variable it", substrs=["item = 3"])

        cit = self.frame().FindVariable("cit")
        self.assertEqual(cit.GetNumChildren(), 1)
        self.assertEqual(cit.GetChildAtIndex(0).GetValueAsSigned(), 3)
        self.expect("frame variable cit", substrs=["item = 3"])
