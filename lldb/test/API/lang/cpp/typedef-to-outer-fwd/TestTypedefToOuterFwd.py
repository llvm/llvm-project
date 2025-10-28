import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCaseTypedefToOuterFwd(TestBase):
    """
    We find a global variable whose type is forward declared
    (whose definition is in either main.o or lib.o). We then
    try to get the 'Ref' typedef nested within that forward
    declared type. This test makes sure we correctly resolve
    this typedef.

    We test this for two cases, where the definition lives
    in main.o or lib.o.
    """

    def check_global_var(self, target, name: str):
        var = target.FindFirstGlobalVariable(name)
        self.assertSuccess(var.GetError(), f"Found {name}")

        var_type = var.GetType()
        self.assertTrue(var_type)

        impl = var_type.GetPointeeType()
        self.assertTrue(impl)

        ref = impl.FindDirectNestedType("Ref")
        self.assertTrue(ref)

        self.assertEqual(ref.GetCanonicalType(), var_type)

    def test(self):
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.check_global_var(target, "gLibExternalDef")
        self.check_global_var(target, "gMainExternalDef")
