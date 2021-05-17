from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

"""
This test ensures that we only create Clang AST nodes in our module AST
when we actually need them.

All tests in this file behave like this:
  1. Use LLDB to do something (expression evaluation, breakpoint setting, etc.).
  2. Check that certain Clang AST nodes were not loaded during the previous
     step.
"""

class TestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
      TestBase.setUp(self)
      # Only build this test once.
      self.build()

    # Clang declaration kind we are looking for.
    class_decl_kind = "CXXRecordDecl"
    # FIXME: This shouldn't be a CXXRecordDecl, but that's how we model
    # structs in Clang.
    struct_decl_kind = "CXXRecordDecl"

    # The decls we use in this program in the format that
    # decl_in_line and decl_completed_in_line expect (which is a pair of
    # node type and the unqualified declaration name.
    struct_first_member_decl = [struct_decl_kind, "StructFirstMember"]
    struct_behind_ptr_decl = [struct_decl_kind, "StructBehindPointer"]
    struct_behind_ref_decl = [struct_decl_kind, "StructBehindRef"]
    struct_member_decl = [struct_decl_kind, "StructMember"]
    some_struct_decl = [struct_decl_kind, "SomeStruct"]
    other_struct_decl = [struct_decl_kind, "OtherStruct"]
    class_in_namespace_decl = [class_decl_kind, "ClassInNamespace"]
    class_we_enter_decl = [class_decl_kind, "ClassWeEnter"]
    class_member_decl = [struct_decl_kind, "ClassMember"]
    class_static_member_decl = [struct_decl_kind, "StaticClassMember"]
    unused_class_member_decl = [struct_decl_kind, "UnusedClassMember"]
    unused_class_member_ptr_decl = [struct_decl_kind, "UnusedClassMemberPtr"]

    def assert_no_decls_loaded(self):
        """
        Asserts that no known declarations in this test are loaded
        into the module's AST.
        """
        self.assert_decl_not_loaded(self.struct_first_member_decl)
        self.assert_decl_not_loaded(self.struct_behind_ptr_decl)
        self.assert_decl_not_loaded(self.struct_behind_ref_decl)
        self.assert_decl_not_loaded(self.struct_member_decl)
        self.assert_decl_not_loaded(self.some_struct_decl)
        self.assert_decl_not_loaded(self.other_struct_decl)
        self.assert_decl_not_loaded(self.class_in_namespace_decl)
        self.assert_decl_not_loaded(self.class_member_decl)
        self.assert_decl_not_loaded(self.class_static_member_decl)
        self.assert_decl_not_loaded(self.unused_class_member_decl)

    def get_ast_dump(self):
        """Returns the dumped Clang AST of the module as a string"""
        res = lldb.SBCommandReturnObject()
        ci = self.dbg.GetCommandInterpreter()
        ci.HandleCommand('target modules dump ast a.out', res)
        self.assertTrue(res.Succeeded())
        return res.GetOutput()

    def decl_in_line(self, line, decl):
        """
        Returns true iff the given line declares the given Clang decl.
        The line is expected to be in the form of Clang's AST dump.
        """
        line = line.rstrip() + "\n"
        decl_kind = "-" + decl[0] + " "
        # Either the decl is somewhere in the line or at the end of
        # the line.
        decl_name = " " + decl[1] + " "
        decl_name_eol = " " + decl[1] + "\n"
        if not decl_kind in line:
          return False
        return decl_name in line or decl_name_eol in line

    def decl_completed_in_line(self, line, decl):
        """
        Returns true iff the given line declares the given Clang decl and
        the decl was completed (i.e., it has no undeserialized declarations
        in it).
        """
        return self.decl_in_line(line, decl) and not "<undeserialized declarations>" in line

    # The following asserts are used for checking if certain Clang declarations
    # were loaded or not since the target was created.

    def assert_decl_loaded(self, decl):
        """
        Asserts the given decl is currently loaded.
        Note: This test is about checking that types/declarations are not
        loaded. If this assert fails it is usually fine to turn it into a
        assert_decl_not_loaded or assert_decl_not_completed assuming LLDB's
        functionality has not suffered by not loading this declaration.
        """
        ast = self.get_ast_dump()
        found = False
        for line in ast.splitlines():
          if self.decl_in_line(line, decl):
            found = True
            self.assertTrue(self.decl_completed_in_line(line, decl),
                            "Should have called assert_decl_not_completed")
        self.assertTrue(found, "Declaration no longer loaded " + str(decl) +
            ".\nAST:\n" + ast)

    def assert_decl_not_completed(self, decl):
        """
        Asserts that the given decl is currently not completed in the module's
        AST. It may be loaded but then can can only contain undeserialized
        declarations.
        """
        ast = self.get_ast_dump()
        found = False
        for line in ast.splitlines():
          error_msg = "Unexpected completed decl: '" + line + "'.\nAST:\n" + ast
          self.assertFalse(self.decl_completed_in_line(line, decl), error_msg)

    def assert_decl_not_loaded(self, decl):
        """
        Asserts that the given decl is currently not loaded in the module's
        AST.
        """
        ast = self.get_ast_dump()
        found = False
        for line in ast.splitlines():
          error_msg = "Unexpected loaded decl: '" + line + "'\nAST:\n" + ast
          self.assertFalse(self.decl_in_line(line, decl), error_msg)


    def clean_setup(self, location):
        """
        Runs to the line with the source line with the given location string
        and ensures that our module AST is empty.
        """
        lldbutil.run_to_source_breakpoint(self,
            "// Location: " + location, lldb.SBFileSpec("main.cpp"))
        # Make sure no declarations are loaded initially.
        self.assert_no_decls_loaded()

    @add_test_categories(["dwarf"])
    def test_arithmetic_expression_in_main(self):
        """ Runs a simple arithmetic expression which should load nothing. """
        self.clean_setup(location="multiple locals function")

        self.expect("expr 1 + (int)2.0", substrs=['(int) $0'])

        # This should not have loaded any decls.
        self.assert_no_decls_loaded()

    @add_test_categories(["dwarf"])
    def test_printing_local_variable_in_other_struct_func(self):
        """
        Prints a local variable and makes sure no unrelated types are loaded.
        """
        self.clean_setup(location="other struct function")

        self.expect("expr other_struct_var", substrs=['(OtherStruct) $0'])
        # The decl we run on was loaded.
        self.assert_decl_loaded(self.other_struct_decl)

        # This should not have loaded anything else.
        self.assert_decl_not_loaded(self.some_struct_decl)
        self.assert_decl_not_loaded(self.class_in_namespace_decl)

    @add_test_categories(["dwarf"])
    def test_printing_struct_with_multiple_locals(self):
        """
        Prints a local variable and checks that we don't load other local
        variables.
        """
        self.clean_setup(location="multiple locals function")

        self.expect("expr struct_var", substrs=['(SomeStruct) $0'])

        # We loaded SomeStruct and its member types for printing.
        self.assert_decl_loaded(self.some_struct_decl)
        self.assert_decl_loaded(self.struct_behind_ptr_decl)
        self.assert_decl_loaded(self.struct_behind_ref_decl)

        # FIXME: We don't use these variables, but we seem to load all local
        # local variables.
        self.assert_decl_not_completed(self.other_struct_decl)
        self.assert_decl_not_completed(self.class_in_namespace_decl)

    @add_test_categories(["dwarf"])
    def test_addr_of_struct(self):
        """
        Prints the address of a local variable (which is a struct).
        """
        self.clean_setup(location="multiple locals function")

        self.expect("expr &struct_var", substrs=['(SomeStruct *) $0'])

        # We loaded SomeStruct.
        self.assert_decl_loaded(self.some_struct_decl)

        # The member declarations should not be completed.
        self.assert_decl_not_completed(self.struct_behind_ptr_decl)
        self.assert_decl_not_completed(self.struct_behind_ref_decl)

        # FIXME: The first member was behind a pointer so it shouldn't be
        # completed. Somehow LLDB really wants to load the first member, so
        # that is why have it defined here.
        self.assert_decl_loaded(self.struct_first_member_decl)

        # FIXME: We don't use these variables, but we seem to load all local
        # local variables.
        self.assert_decl_not_completed(self.other_struct_decl)
        self.assert_decl_not_completed(self.class_in_namespace_decl)

    @add_test_categories(["dwarf"])
    def test_class_function_access_member(self):
        self.clean_setup(location="class function")

        self.expect("expr member", substrs=['(ClassMember) $0'])

        # We loaded the current class we touched.
        self.assert_decl_loaded(self.class_we_enter_decl)
        # We loaded the unused members of this class.
        self.assert_decl_loaded(self.unused_class_member_decl)
        self.assert_decl_not_completed(self.unused_class_member_ptr_decl)
        # We loaded the member we used.
        self.assert_decl_loaded(self.class_member_decl)
        # We didn't load the type of the unused static member.
        self.assert_decl_not_completed(self.class_static_member_decl)

        # This should not have loaded anything else.
        self.assert_decl_not_loaded(self.other_struct_decl)
        self.assert_decl_not_loaded(self.some_struct_decl)
        self.assert_decl_not_loaded(self.class_in_namespace_decl)

