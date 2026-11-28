
# Tests correct behaviour of GetNumChildren and GetChildAtIndex
# for references to struct

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestReferenceToStruct(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        frame = self.frame()

        # A reference to an aggregate is displayed transparently, so it must
        # report the same number of children as the referent.

        # GetNumChildren should return 0 for reference to empty struct

        e_ref = frame.FindVariable("e_ref")
        self.assertEqual(e_ref.GetNumChildren(), 0)

        # GetNumChildren should return 1 for reference to struct containing
        # a single child

        s_ref = frame.FindVariable("s_ref")
        self.assertEqual(s_ref.GetNumChildren(), 1)
        child = s_ref.GetChildAtIndex(0)
        self.assertTrue(child.IsValid())
        self.assertEqual(child.GetName(), "x")

        # GetNumChildren should return 2 for reference to struct containing
        # two children

        t_ref = frame.FindVariable("t_ref")
        self.assertEqual(t_ref.GetNumChildren(), 2)

        child1 = t_ref.GetChildAtIndex(0)
        self.assertTrue(child1.IsValid())
        self.assertEqual(child1.GetName(), "x")

        child2 = t_ref.GetChildAtIndex(1)
        self.assertTrue(child2.IsValid())
        self.assertEqual(child2.GetName(), "y")
