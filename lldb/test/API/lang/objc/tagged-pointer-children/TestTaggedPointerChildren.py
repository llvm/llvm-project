import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TaggedPointerChildrenTestCase(TestBase):
    def is_tagged(self, valobj):
        """Return True if valobj holds an Objective-C tagged pointer."""
        res = lldb.SBCommandReturnObject()
        ci = self.dbg.GetCommandInterpreter()
        ci.HandleCommand(
            "language objc tagged-pointer info %s" % valobj.GetValue(), res
        )
        return res.Succeeded() and "is tagged" in res.GetOutput()

    def assert_no_unreadable_children(self, valobj):
        """Recursively assert no child of valobj fails with a memory-read error."""
        for i in range(valobj.GetNumChildren()):
            child = valobj.GetChildAtIndex(i)
            err = child.GetError()
            self.assertFalse(
                err.Fail() and "read memory" in (err.GetCString() or ""),
                "child '%s' of '%s' has a memory-read error: %s"
                % (child.GetName(), valobj.GetName(), err.GetCString()),
            )
            self.assert_no_unreadable_children(child)

    @skipUnlessDarwin
    @skipIf(archs=["i386", "i686"])
    def test(self):
        """
        Test that Objective-C tagged pointers (inline values) do not present phantom,
        memory-backed children.
        """
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m")
        )
        frame = thread.GetSelectedFrame()

        tagged = frame.FindVariable("tagged")
        self.assertTrue(tagged.IsValid(), "found 'tagged'")

        # If this platform does not represent a single-index NSIndexSet as a
        # tagged pointer there is nothing to test.
        if not self.is_tagged(tagged):
            self.skipTest("NSIndexSet is not a tagged pointer on this platform")

        # The tagged pointer is still summarized.
        self.assertIsNotNone(tagged.GetSummary())
        self.assertIn("index", tagged.GetSummary())

        # But it must not present any unresolvable children.
        self.assertEqual(
            tagged.GetNumChildren(),
            0,
            "tagged pointer must not have children",
        )
        self.assertFalse(tagged.MightHaveChildren())
        self.assert_no_unreadable_children(tagged)

        # Sanity check.
        heap = frame.FindVariable("heap")
        self.assertTrue(heap.IsValid(), "found 'heap'")
        self.assertFalse(self.is_tagged(heap), "'heap' is a real object")
        self.assertGreater(heap.GetNumChildren(), 0, "real object still has children")
        self.assert_no_unreadable_children(heap)

    @skipUnlessDarwin
    @skipIf(archs=["i386", "i686"])
    def test_registered_synthetic_takes_precedence(self):
        """
        A synthetic child provider registered for the type must still fire for
        a tagged pointer. ObjCLanguage's hardcoded "no children" provider is
        only a fallback and must be consulted after any registered synthetic.
        """
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m")
        )
        frame = thread.GetSelectedFrame()

        tagged = frame.FindVariable("tagged")
        self.assertTrue(tagged.IsValid(), "found 'tagged'")
        if not self.is_tagged(tagged):
            self.skipTest("NSIndexSet is not a tagged pointer on this platform")

        # Without a registered synthetic, the hardcoded fallback reports no
        # children for the tagged pointer.
        self.assertEqual(tagged.GetNumChildren(), 0)

        # Register a synthetic child provider for NSIndexSet.
        import os

        provider = os.path.join(os.path.dirname(__file__), "tagged_synth_provider.py")
        self.runCmd("command script import " + provider)
        self.runCmd(
            'type synthetic add -x "NSIndexSet" '
            "-l tagged_synth_provider.TaggedSyntheticProvider"
        )
        self.addTearDownHook(lambda: self.runCmd("type synthetic clear", check=False))

        # The registered provider must now win over the hardcoded fallback.
        tagged = frame.FindVariable("tagged")
        self.assertEqual(
            tagged.GetNumChildren(),
            1,
            "registered synthetic must take precedence over the hardcoded "
            "tagged-pointer fallback",
        )
        self.assertTrue(tagged.MightHaveChildren())
        child = tagged.GetChildAtIndex(0)
        self.assertEqual(child.GetName(), "synthetic_child")
        self.assertEqual(child.GetValueAsSigned(), 42)
