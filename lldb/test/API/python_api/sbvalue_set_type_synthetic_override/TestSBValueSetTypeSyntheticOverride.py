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
        foo = lldb.SBTypeSynthetic.CreateWithClassName(f"foo_bar_synths.FooSynthetic")
        bar = lldb.SBTypeSynthetic.CreateWithClassName(f"foo_bar_synths.BarSynthetic")

        # Target the static (non synthetic) ValueObject
        static = value.GetNonSyntheticValue()

        impl_before = static.GetSyntheticValue().GetTypeSyntheticImplementation()
        if not before:
            self.assertIsNone(impl_before)
        else:
            self.assertIsNotNone(impl_before)
            self.assertEqual(type(impl_before).__name__, before)

        static.SetTypeSynthetic(bar)
        self.assertEqual(static.GetTypeSynthetic(), bar)

        impl_after = static.GetSyntheticValue().GetTypeSyntheticImplementation()
        self.assertIsNotNone(impl_after)
        self.assertEqual(type(impl_after).__name__, "BarSynthetic")

        # Target the ValueObjectSynthetic of the original ValueObject
        synth = value.GetSyntheticValue()

        synth.SetTypeSynthetic(foo)
        self.assertEqual(synth.GetTypeSynthetic(), foo)

        # Even though the synthetic child provider choice of 'synth' was changed
        # that does not retroactively alter the frontend it was created with when
        # its parent's synthetic child provider choice was overridden above.
        impl_after = synth.GetSyntheticValue().GetTypeSyntheticImplementation()
        self.assertIsNotNone(impl_after)
        self.assertEqual(type(impl_after).__name__, "BarSynthetic")
