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
        self.assertEqual(foos.GetNumChildren(), 2)

        foos_impl = foos.GetTypeSyntheticImplementation()
        self.assertEqual(type(foos_impl).__name__, "FooHandleArraySynthetic")

        # And it correcly converts to typed children
        foos0 = foos.GetChildAtIndex(0)
        self.assertTrue(foos0.IsValid())
        self.assertEqual(foos0.GetType(), target.FindFirstType("Foo").GetPointerType())

        # Same with "bars"
        bars = info.GetChildMemberWithName("bars")
        self.assertTrue(bars.IsValid())
        self.assertTrue(bars.IsSynthetic())
        self.assertEqual(bars.GetNumChildren(), 1)

        bars_impl = bars.GetTypeSyntheticImplementation()
        self.assertEqual(type(bars_impl).__name__, "BarHandleArraySynthetic")

        bars0 = bars.GetChildAtIndex(0)
        self.assertTrue(bars0.IsValid())
        self.assertEqual(bars0.GetType(), target.FindFirstType("Bar").GetPointerType())
