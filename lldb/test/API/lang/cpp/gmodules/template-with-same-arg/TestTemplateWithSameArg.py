"""
Tests the scenario where we evaluate expressions
of two types in different modules that reference
a class template instantiated with the same
template argument.

Note that,
1. Since the decls originate from modules, LLDB
   marks them as such and Clang doesn't create
   a LookupPtr map on the corresponding DeclContext.
   This prevents regular DeclContext::lookup from
   succeeding.
2. Because we reference the same class template
   from two different modules we get a redeclaration
   chain for the class's ClassTemplateSpecializationDecl.
   The importer will import all FieldDecls into the
   same DeclContext on the redeclaration chain. If
   we don't do the bookkeeping correctly we end up
   with duplicate decls on the same DeclContext leading
   to crashes down the line.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestTemplateWithSameArg(TestBase):

    def setUp(self):
        TestBase.setUp(self)
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.cpp")

    @add_test_categories(["gmodules"])
    @skipIf(bugnumber='rdar://96581048')
    def test_same_template_arg(self):
        lldbutil.run_to_source_breakpoint(self, "Break here", self.main_source_file)

        self.expect_expr("FromMod1", result_type="ClassInMod1", result_children=[
                ValueCheck(name="VecInMod1", children=[
                            ValueCheck(name="Member", value="137")
                    ])
            ])

        self.expect_expr("FromMod2", result_type="ClassInMod2", result_children=[
                ValueCheck(name="VecInMod2", children=[
                            ValueCheck(name="Member", value="42")
                    ])
            ])

    @add_test_categories(["gmodules"])
    @skipIf(bugnumber='rdar://96581048')
    def test_duplicate_decls(self):
        lldbutil.run_to_source_breakpoint(self, "Break here", self.main_source_file)

        self.expect_expr("(intptr_t)&FromMod1 + (intptr_t)&FromMod2")

        # Make sure we only have a single 'Member' decl on the AST
        self.filecheck("target module dump ast", __file__)
# CHECK:      ClassTemplateSpecializationDecl {{.*}} imported in Module2 struct ClassInMod3 definition
# CHECK-NEXT: |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial
# CHECK-NEXT: | |-DefaultConstructor exists trivial needs_implicit
# CHECK-NEXT: | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
# CHECK-NEXT: | |-MoveConstructor exists simple trivial needs_implicit
# CHECK-NEXT: | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
# CHECK-NEXT: | |-MoveAssignment exists simple trivial needs_implicit
# CHECK-NEXT: | `-Destructor simple irrelevant trivial needs_implicit
# CHECK-NEXT: |-TemplateArgument type 'int'
# CHECK-NEXT: | `-BuiltinType {{.*}} 'int'
# CHECK-NEXT: `-FieldDecl {{.*}} imported in Module2 Member 'int'
