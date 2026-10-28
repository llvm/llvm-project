import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class DynamicValueComplexTypename(TestBase):
    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler="clang", compiler_version=["<", "24.0"])  # __clang_vtable
    @expectedFailureAll(debug_info=["pdb"])  # Not implemented
    @expectedFailureAll(oslist=["windows"])  # Not Itanium ABI
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        self.checkVarType("simple", "Simple")
        self.checkVarType("templated", "Templated<")
        self.checkVarType("complicated", "Complicated<")
        self.checkVarType("evil", "Evil<")

    def checkVarType(self, name, type_prefix):
        var = self.frame().FindVariable(name)
        self.assertTrue(var.IsValid())

        # Check if we recognized the dynamic type. Most names are unreasonably
        # complex to match against exactly, but all should have a sane prefix
        type_name = var.GetType().GetName()
        self.assertTrue(
            type_name.startswith(type_prefix),
            f"Expected '{type_prefix}' prefix, got '{type_name}'",
        )

        # Check if we resolved the compiler type too
        marker = var.Dereference().GetChildAtIndex(0)
        self.assertEqual(marker.GetName(), f"is_{name}")
        self.assertEqual(marker.GetValueAsUnsigned(), 1)
