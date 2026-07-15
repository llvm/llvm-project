import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )
        self.runCmd("command script import foo_bar_synths.py")

        frame = thread.GetFrameAtIndex(0)

        # CXX runtime synthetic can be overridden
        vec = frame.FindVariable("vec")
        self.assertTrue(vec.IsSynthetic())
        self.checkOverride(vec, before=None)

        # Python synthetic can be overridden
        foo = frame.FindVariable("foo")
        self.assertTrue(foo.IsSynthetic())
        self.checkOverride(foo, before="FooSynthetic")

        # No synthetic can be overridden
        bar = frame.FindVariable("bar")
        self.assertFalse(bar.IsSynthetic())
        self.checkOverride(bar, before=None)

    def checkOverride(self, value, before):
        bar = lldb.SBTypeSynthetic.CreateWithClassName(f"foo_bar_synths.BarSynthetic")

        impl_before = value.GetTypeSyntheticImplementation()

        if not before:
            self.assertIsNone(impl_before)
        else:
            self.assertIsNotNone(impl_before)
            self.assertEqual(type(impl_before).__name__, before)

        value.SetTypeSynthetic(bar)
        self.assertEqual(value.GetTypeSynthetic(), bar)

        impl_after = value.GetTypeSyntheticImplementation()
        self.assertIsNotNone(impl_after)
        self.assertEqual(type(impl_after).__name__, "BarSynthetic")
