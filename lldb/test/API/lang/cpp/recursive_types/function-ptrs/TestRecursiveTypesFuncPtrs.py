"""
Test that LLDB can complete and evaluate recursive types without infinitely
recursing. Each type reaches itself through a function pointer that returns a
pointer to the type.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build_and_run()

        # Recursion through the named struct 's1'. Completing 't1' and walking
        # one level into its members must not recurse infinitely.
        self.expect_var_path("p1", type="t1_ptr")
        self.expect_expr("p1", result_type="t1_ptr")
        self.expect_expr("*p1", result_type="t1")
        self.expect_expr("p1->s", result_type="s1 *")

        # Recursion through an anonymous struct member.
        self.expect_var_path("p2", type="t2_ptr")
        self.expect_expr("p2", result_type="t2_ptr")
        self.expect_expr("*p2", result_type="t2")
