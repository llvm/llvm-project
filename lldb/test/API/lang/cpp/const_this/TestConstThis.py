import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def run_class_tests(self):
        # Expression not referencing context class.
        self.expect_expr("1 + 1", result_type="int", result_value="2")
        # Referencing context class.
        # FIXME: This and the expression below should return const types.
        self.expect_expr("member", result_type="int", result_value="3")
        # Check the type of context class.
        self.expect_expr("this", result_type="ContextClass *")

    def test_member_func(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break in function in class.", lldb.SBFileSpec("main.cpp")
        )
        self.run_class_tests()

    def test_templated_member_func(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self,
            "// break in templated function in class.",
            lldb.SBFileSpec("main.cpp"),
        )
        self.run_class_tests()

    def run_template_class_tests(self):
        # Expression not referencing context class.
        self.expect_expr("1 + 1", result_type="int", result_value="2")
        # Referencing context class.
        # FIXME: This and the expression below should return const types.
        self.expect_expr("member", result_type="int", result_value="4")
        # Check the type of context class.
        self.expect_expr("this", result_type="TemplatedContextClass<int> *")

    def test_template_member_func(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self,
            "// break in function in templated class.",
            lldb.SBFileSpec("main.cpp"),
        )
        self.run_template_class_tests()

    def test_template_templated_member_func(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self,
            "// break in templated function in templated class.",
            lldb.SBFileSpec("main.cpp"),
        )
        self.run_template_class_tests()
