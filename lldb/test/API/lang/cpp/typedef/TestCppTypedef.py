"""
Test typedef types.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test import decorators


class TestCppTypedef(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_typedef(self):
        """
        Test that we retrieve typedefed types correctly
        """

        self.build()
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )


        # First of all, check that we can get a typedefed type correctly in a simple case.
        expr_result = self.expect_expr(
            "(GlobalTypedef)s",
            result_type="GlobalTypedef",
            result_children=[ValueCheck(value="0.5")],
        )

        # The type should be a typedef.
        typedef_type = expr_result.GetType()
        self.assertTrue(typedef_type.IsValid())
        self.assertTrue(typedef_type.IsTypedefType())

        # The underlying type should be S<float>.
        typedefed_type = typedef_type.GetTypedefedType()
        self.assertTrue(typedefed_type.IsValid())
        self.assertEqual(typedefed_type.GetName(), "S<float>")


        # Check that we can get a typedefed type correctly in the case
        # when an elaborated type is created during the parsing
        expr_result = self.expect_expr(
            "(GlobalTypedef::V)s.value", result_type="GlobalTypedef::V"
        )

        # The type should be a typedef.
        typedef_type = expr_result.GetType()
        self.assertTrue(typedef_type.IsValid())
        self.assertTrue(typedef_type.IsTypedefType())

        # The underlying type should be float.
        typedefed_type = typedef_type.GetTypedefedType()
        self.assertTrue(typedefed_type.IsValid())
        self.assertEqual(typedefed_type.GetName(), "float")


        # Try accessing a typedef inside a namespace.
        self.expect_expr(
            "(ns::NamespaceTypedef)s", result_children=[ValueCheck(value="0.5")]
        )


        # Try accessing a typedef inside a struct/class.
        # FIXME: This doesn't actually work. StructTypedef just gets injected
        # by the local variable in the expression evaluation context.
        self.expect_expr(
            "(ST::StructTypedef)s", result_children=[ValueCheck(value="0.5")]
        )
        # This doesn't work for the reason above. There is no local variable
        # injecting OtherStructTypedef so we will actually error here.
        self.expect(
            "expression -- (NonLocalVarStruct::OtherStructTypedef)1",
            error=True,
            substrs=["no member named 'OtherStructTypedef' in 'NonLocalVarStruct'"],
        )


        # Check the generated Clang AST.
        self.filecheck("image dump ast a.out", __file__, "--strict-whitespace")
# CHECK:      {{^}}|-TypedefDecl {{.*}} GlobalTypedef 'S<float>'
# CHECK:      {{^}}|-NamespaceDecl {{.*}} ns
# CHECK-NEXT: {{^}}| `-TypedefDecl {{.*}} NamespaceTypedef 'S<float>'
# CHECK:      {{^}}|-CXXRecordDecl {{.*}} struct ST definition
# CHECK:      {{^}}| `-TypedefDecl {{.*}} StructTypedef 'S<float>'
