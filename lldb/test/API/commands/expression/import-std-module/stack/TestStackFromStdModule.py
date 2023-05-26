"""
Tests std::stack functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestStack(TestBase):
    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    @skipIfLinux  # Declaration in some Linux headers causes LLDB to crash.
    @skipIf(bugnumber="rdar://97622854")
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.import-std-module true")

        # Test std::stack functionality with a std::deque.
        stack_type = "std::stack<C>"
        size_type = stack_type + "::size_type"

        self.expect_expr("s_deque", result_type=stack_type)
        self.expect("expr s_deque.pop()")
        self.expect("expr s_deque.push({4})")
        self.expect_expr("s_deque.size()", result_type=size_type, result_value="3")
        self.expect_expr("s_deque.top().i", result_type="int", result_value="4")
        self.expect("expr s_deque.emplace(5)")
        self.expect_expr("s_deque.top().i", result_type="int", result_value="5")

        # Test std::stack functionality with a std::vector.
        stack_type = "std::stack<C, std::vector<C> >"
        size_type = stack_type + "::size_type"

        self.expect_expr("s_vector", result_type=stack_type)
        self.expect("expr s_vector.pop()")
        self.expect("expr s_vector.push({4})")
        self.expect_expr("s_vector.size()", result_type=size_type, result_value="3")
        self.expect_expr("s_vector.top().i", result_type="int", result_value="4")
        self.expect("expr s_vector.emplace(5)")
        self.expect_expr("s_vector.top().i", result_type="int", result_value="5")

        # Test std::stack functionality with a std::list.
        stack_type = "std::stack<C, std::list<C> >"
        size_type = stack_type + "::size_type"
        self.expect_expr("s_list", result_type=stack_type)
        self.expect("expr s_list.pop()")
        self.expect("expr s_list.push({4})")
        self.expect_expr("s_list.size()", result_type=size_type, result_value="3")
        self.expect_expr("s_list.top().i", result_type="int", result_value="4")
        self.expect("expr s_list.emplace(5)")
        self.expect_expr("s_list.top().i", result_type="int", result_value="5")
