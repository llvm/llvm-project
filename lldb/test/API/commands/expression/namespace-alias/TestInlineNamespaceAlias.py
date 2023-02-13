"""
Test that we correctly handle namespace
expression evaluation through namespace
aliases.
"""

import lldb

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestInlineNamespace(TestBase):
    def do_test(self, params):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "return A::B::C::a", lldb.SBFileSpec("main.cpp"))

        self.expect_expr("A::C::a", result_type="int", result_value="-1")
        self.expect_expr("A::D::a", result_type="int", result_value="-1")

        self.expect_expr("A::C::func()", result_type="int", result_value="0")
        self.expect_expr("A::D::func()", result_type="int", result_value="0")

        self.expect_expr("E::C::a", result_type="int", result_value="-1")
        self.expect_expr("E::D::a", result_type="int", result_value="-1")
        self.expect_expr("F::a", result_type="int", result_value="-1")
        self.expect_expr("G::a", result_type="int", result_value="-1")

    @skipIf(debug_info=no_match(["dsym"]))
    def test_dsym(self):
        self.do_test({})

    @skipIf(debug_info="dsym")
    def test_dwarf(self):
        self.do_test(dict(CFLAGS_EXTRAS="-gdwarf-5 -gpubnames"))
