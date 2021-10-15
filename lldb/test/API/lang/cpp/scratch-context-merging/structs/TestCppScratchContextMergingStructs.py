"""
This tests LLDB's ability to merge structs into the shared per-target Clang
ASTContext.

This just focuses on indirect imports (i.e., a declaration gets imported from
the lldb::Module AST into the expression AST and then the declaration gets
imported to the scratch AST because it is part of the ValueObject type of the
result) and direct imports (i.e., a declaration gets directly imported from a
lldb::Module AST to the scratch AST, e.g., via 'frame var').
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def common_setup(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

    def do_pass(self, kind, var, expected_type, expected_children):
        if kind == "expression":
            self.expect_expr(
                var, result_type=expected_type, result_children=expected_children
            )
        elif kind == "path":
            self.expect_var_path(var, type=expected_type, children=expected_children)
        else:
            self.fail("Unknown var evaluation kind: " + var)

    def pull_in_and_merge(self, var, type, children):
        """
        Pulls in the specified variable into the scratch AST. Afterwards tries
        merging the declaration. The method of pulling the declaration into the
        scratch AST is defined by the first_pass/second_pass instance variables.
        """

        # This pulls in the declaration into the scratch AST.
        self.do_pass(self.first_pass, var, type, children)
        # This pulls in the declaration a second time and forces us to merge with
        # the existing declaration (or reuse the existing declaration).
        self.do_pass(self.second_pass, var, type, children)

    def do_tests(self):
        """ Just forwards all the variables/types/childrens to pull_in_and_merge. """
        self.pull_in_and_merge(
            "decl_in_func", type="DeclInFunc", children=[ValueCheck(name="member")]
        )
        self.pull_in_and_merge(
            "top_level_struct",
            type="TopLevelStruct",
            children=[ValueCheck(name="member")],
        )
        self.pull_in_and_merge(
            "inner_struct",
            type="OuterStruct::InnerStruct",
            children=[ValueCheck(name="member")],
        )
        self.pull_in_and_merge(
            "typedef_struct",
            type="TypedefStruct",
            children=[ValueCheck(name="member")],
        )
        self.pull_in_and_merge(
            "namespace_struct",
            type="NS::NamespaceStruct",
            children=[ValueCheck(name="member")],
        )
        self.pull_in_and_merge(
            "unnamed_namespace_struct",
            type="UnnamedNamespaceStruct",
            children=[ValueCheck(name="member")],
        )
        self.pull_in_and_merge(
            "extern_c_struct",
            type="ExternCStruct",
            children=[ValueCheck(name="member")],
        )

    @no_debug_info_test
    def test_direct_and_indirect(self):
        """
        First variable paths pull in a declaration directly. Then the expression
        evaluator pulls the declaration in indirectly.
        """
        self.common_setup()
        self.first_pass = "path"
        self.second_pass = "expression"
        self.do_tests()

    @no_debug_info_test
    def test_indirect_and_indirect(self):
        """
        The expression evaluator pulls in the declaration indirectly and then
        repeat that process.
        """
        self.common_setup()
        self.first_pass = "expression"
        self.second_pass = "expression"
        self.do_tests()

    @no_debug_info_test
    def test_indirect_and_direct(self):
        """
        The expression evaluator pulls in the declaration indirectly and then
        variable paths pull it in directly.
        """
        self.common_setup()
        self.first_pass = "expression"
        self.second_pass = "path"
        self.do_tests()

    @no_debug_info_test
    def test_direct_and_direct(self):
        """
        Variable paths pulls in the declaration indirectly and then repeat that
        process.
        """
        self.common_setup()
        self.first_pass = "path"
        self.second_pass = "path"
        self.do_tests()
