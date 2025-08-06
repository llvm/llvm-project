"""
Tests std::queue functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestQueue(TestBase):
    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    @skipIf(
        compiler="clang",
        compiler_version=["<", "17.0"],
    )
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.import-std-module true")

        if self.expectedCompiler(["clang"]) and self.expectedCompilerVersion(
            [">", "16.0"]
        ):
            queue_type = "std::queue<C>"
        else:
            queue_type = "std::queue<C, std::deque<C, std::allocator<C> > >"

        size_type = "size_type"
        value_type = "value_type"

        # Test std::queue functionality with a std::deque.
        self.expect_expr(
            "q_deque",
            result_type=queue_type,
            result_children=[ValueCheck(children=[ValueCheck(value="1")])],
        )
        self.expect("expr q_deque.pop()")
        self.expect("expr q_deque.push({4})")
        self.expect_expr("q_deque.size()", result_type=size_type, result_value="1")
        self.expect_expr("q_deque.front()", result_type=value_type)
        self.expect_expr("q_deque.back()", result_type=value_type)
        self.expect_expr("q_deque.front().i", result_type="int", result_value="4")
        self.expect_expr("q_deque.back().i", result_type="int", result_value="4")
        self.expect_expr("q_deque.empty()", result_type="bool", result_value="false")
        self.expect("expr q_deque.pop()")
        self.expect("expr q_deque.emplace(5)")
        self.expect_expr("q_deque.front().i", result_type="int", result_value="5")

        # Test std::queue functionality with a std::list.
        if self.expectedCompiler(["clang"]) and self.expectedCompilerVersion(
            [">", "16.0"]
        ):
            queue_type = "std::queue<C, std::list<C> >"
        else:
            queue_type = "std::queue<C, std::list<C, std::allocator<C> > >"

        self.expect_expr(
            "q_list",
            result_type=queue_type,
            result_children=[ValueCheck(children=[ValueCheck(value="1")])],
        )

        self.expect("expr q_list.pop()")
        self.expect("expr q_list.push({4})")
        self.expect_expr("q_list.size()", result_type=size_type, result_value="1")
        self.expect_expr("q_list.front()", result_type=value_type)
        self.expect_expr("q_list.back()", result_type=value_type)
        self.expect_expr("q_list.front().i", result_type="int", result_value="4")
        self.expect_expr("q_list.back().i", result_type="int", result_value="4")
        self.expect_expr("q_list.empty()", result_type="bool", result_value="false")
        self.expect("expr q_list.pop()")
        self.expect("expr q_list.emplace(5)")
        self.expect_expr("q_list.front().i", result_type="int", result_value="5")
