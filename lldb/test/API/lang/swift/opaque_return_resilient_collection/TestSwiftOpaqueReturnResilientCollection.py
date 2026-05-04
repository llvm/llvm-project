import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftOpaqueReturnResilientCollection(TestBase):
    @swiftTest
    @skipIfWindows
    def test(self):
        self.build()
        self.runCmd("settings set symbols.swift-enable-ast-context false")
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"),
            extra_images=['OpaqueLib'])

        frame = thread.GetSelectedFrame()
        manager = frame.FindVariable("manager")
        items = manager.GetChildMemberWithName("items")
        self.assertTrue(items.IsValid(), "items should be valid")
        self.assertIn("Creator", items.GetTypeName())

        first = items.GetChildAtIndex(0)
        self.assertTrue(first.IsValid(), "items[0] should be valid")
        self.assertIn("Creator", first.GetTypeName())

        name = first.GetChildMemberWithName("name")
        self.assertTrue(name.IsValid(), "name child should be valid")
        self.assertEqual(name.GetSummary(), '"one"')
