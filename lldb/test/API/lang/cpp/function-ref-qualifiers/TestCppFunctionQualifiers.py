"""
Tests that C++ expression evaluation can
disambiguate between rvalue and lvalue
reference-qualified functions.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "Break here", lldb.SBFileSpec("main.cpp"))

        # const lvalue
        self.expect_expr("const_foo.func()", result_type="uint32_t", result_value="0")

        # const rvalue
        self.expect_expr("static_cast<Foo const&&>(Foo{}).func()",
                         result_type="int64_t", result_value="1")

        # non-const lvalue
        self.expect_expr("foo.func()", result_type="uint32_t", result_value="2")

        # non-const rvalue
        self.expect_expr("Foo{}.func()", result_type="int64_t", result_value="3")

        self.filecheck("target modules dump ast", __file__)
        # CHECK:      |-CXXMethodDecl {{.*}} func 'uint32_t () const &'
        # CHECK-NEXT: | `-AsmLabelAttr {{.*}}
        # CHECK-NEXT: |-CXXMethodDecl {{.*}} func 'int64_t () const &&'
        # CHECK-NEXT: | `-AsmLabelAttr {{.*}}
        # CHECK-NEXT: |-CXXMethodDecl {{.*}} func 'uint32_t () &'
        # CHECK-NEXT: | `-AsmLabelAttr {{.*}}
        # CHECK-NEXT: `-CXXMethodDecl {{.*}} func 'int64_t () &&'
        # CHECK-NEXT:   `-AsmLabelAttr {{.*}}
