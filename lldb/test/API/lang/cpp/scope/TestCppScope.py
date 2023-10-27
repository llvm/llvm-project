import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24764")
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        # Test that global variables contain the right scope operators.
        global_vars = self.frame().GetVariables(False, False, True, False)
        # ManualDWARFIndex using NameToDIE does not sort alphabetically.
        global_var_names = sorted([v.GetName() for v in global_vars])
        expected_var_names = ["::a", "A::a", "B::a", "C::a"]
        self.assertEqual(global_var_names, expected_var_names)

        # Test lookup in scopes.
        self.expect_expr("A::a", result_value="1111")
        self.expect_expr("B::a", result_value="2222")
        self.expect_expr("C::a", result_value="3333")
        self.expect_expr("::a", result_value="4444")
        # Check that lookup without scope returns the same result.
        self.expect_expr("a", result_value="4444")
