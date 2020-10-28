"""
Test that we can retrieve typedefed types correctly
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

        # Build and run until the breakpoint
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source_file)

        # Get the current frame
        frame = thread.GetSelectedFrame()

        # First of all, check that we can get a typedefed type correctly in a simple case

        expr_result = self.expect_expr("(SF)s", result_children=[ValueCheck(value="0.5")])
        self.expect_expr("(ns::SF)s", result_children=[ValueCheck(value="0.5")])
        self.expect_expr("(ST::SF)s", result_children=[ValueCheck(value="0.5")])

        self.filecheck("image dump ast a.out", __file__, "--strict-whitespace")
# CHECK:      {{^}}|-TypedefDecl {{.*}} SF 'S<float>'
# CHECK:      {{^}}|-NamespaceDecl {{.*}} ns
# CHECK-NEXT: {{^}}| `-TypedefDecl {{.*}} SF 'S<float>'
# CHECK:      {{^}}`-CXXRecordDecl {{.*}} struct ST definition
# CHECK:      {{^}}  `-TypedefDecl {{.*}} SF 'S<float>'

        typedef_type = expr_result.GetType();
        self.assertTrue(typedef_type.IsValid(), "Can't get `SF` type of evaluated expression")
        self.assertTrue(typedef_type.IsTypedefType(), "Type `SF` should be a typedef")

        typedefed_type = typedef_type.GetTypedefedType()
        self.assertTrue(typedefed_type.IsValid(), "Can't get `SF` typedefed type")
        self.assertEqual(typedefed_type.GetName(), "S<float>", "Got invalid `SF` typedefed type")

        # Check that we can get a typedefed type correctly in the case
        # when an elaborated type is created during the parsing

        expr_result = frame.EvaluateExpression("(SF::V)s.value")
        self.assertTrue(expr_result.IsValid(), "Expression failed with: " + str(expr_result.GetError()))

        typedef_type = expr_result.GetType();
        self.assertTrue(typedef_type.IsValid(), "Can't get `SF::V` type of evaluated expression")
        self.assertTrue(typedef_type.IsTypedefType(), "Type `SF::V` should be a typedef")

        typedefed_type = typedef_type.GetTypedefedType()
        self.assertTrue(typedefed_type.IsValid(), "Can't get `SF::V` typedefed type")
        self.assertEqual(typedefed_type.GetName(), "float", "Got invalid `SF::V` typedefed type")
