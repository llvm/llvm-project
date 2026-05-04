import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftOpaqueReturnResilient(TestBase):
    @swiftTest
    @skipIfWindows
    def test(self):
        self.build()
        self.runCmd("settings set symbols.swift-enable-ast-context false")
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"),
            extra_images=['OpaqueLib'])

        frame = thread.GetSelectedFrame()
        variable = frame.FindVariable("variable").GetDynamicValue(
            lldb.eDynamicCanRunTarget)
        self.assertTrue(variable.IsValid(), "variable should be valid")
        self.assertIn("Creator", variable.GetTypeName())

        name = variable.GetChildMemberWithName("name")
        self.assertTrue(name.IsValid(), "name child should be valid")
        self.assertEqual(name.GetSummary(), '"alpha"')
