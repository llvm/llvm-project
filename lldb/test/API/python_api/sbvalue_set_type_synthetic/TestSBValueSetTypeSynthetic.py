import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from typing import Union


class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        target, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )
        self.runCmd("command script import library_support.py")

        foo_t = target.FindFirstType("Foo")
        bar_t = target.FindFirstType("Bar")

        info = thread.GetFrameAtIndex(0).FindVariable("info")
        self.assertTrue(info.IsValid())
        self.assertTrue(info.IsSynthetic())

        # Synthetic attached to "info" can be interrogated
        impl = info.GetTypeSyntheticImplementation()
        self.assertEqual(type(impl).__name__, "SessionInfoSynthetic")

        # A synthetic was manually attached to "foos"
        foos = info.GetChildMemberWithName("foos")
        self.assertTrue(foos.IsValid())
        self.assertTrue(foos.IsSynthetic())

        foos_impl = foos.GetTypeSyntheticImplementation()
        self.assertEqual(type(foos_impl).__name__, "FooHandleArraySynthetic")

        # And it correcly imbues it's "data" member with type information
        foos_data = foos.GetChildMemberWithName("data")
        self.assertTrue(foos_data.IsValid())
        self.assertEqual(
            foos_data.GetType(), foo_t.GetPointerType().GetArrayType(2).GetPointerType()
        )

        foos_data0 = foos_data.Dereference().GetChildAtIndex(0).Dereference()
        self.assertTrue(foos_data0.IsValid())
        self.assertEqual(foos_data0.GetType(), foo_t)

        # Same with "bars"
        bars = info.GetChildMemberWithName("bars")
        self.assertTrue(bars.IsValid())
        self.assertTrue(bars.IsSynthetic())

        bars_impl = bars.GetTypeSyntheticImplementation()
        self.assertEqual(type(bars_impl).__name__, "BarHandleArraySynthetic")

        bars_data = bars.GetChildMemberWithName("data")
        self.assertTrue(bars_data.IsValid())
        self.assertEqual(
            bars_data.GetType(), bar_t.GetPointerType().GetArrayType(1).GetPointerType()
        )

        bars_data0 = bars_data.Dereference().GetChildAtIndex(0).Dereference()
        self.assertTrue(bars_data0.IsValid())
        self.assertTrue(bars_data0.GetType(), bar_t)
