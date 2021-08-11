import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def common_setup(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))

    def test_call_on_base(self):
        self.common_setup()
        self.expect_expr("base_with_dtor.foo()", result_type="int", result_value="1")
        self.expect_expr("base_without_dtor.foo()", result_type="int", result_value="2")

    def test_call_on_derived(self):
        self.common_setup()
        self.expect_expr("derived_with_dtor.foo()", result_type="int", result_value="3")
        self.expect_expr("derived_without_dtor.foo()", result_type="int", result_value="4")
        self.expect_expr("derived_with_base_dtor.foo()", result_type="int", result_value="5")
        self.expect_expr("derived_with_dtor_but_no_base_dtor.foo()", result_type="int", result_value="6")

    def test_call_on_derived_as_base(self):
        self.common_setup()
        self.expect_expr("derived_with_dtor_as_base.foo()", result_type="int", result_value="3")
        self.expect_expr("derived_without_as_base.foo()", result_type="int", result_value="4")
        self.expect_expr("derived_with_base_dtor_as_base.foo()", result_type="int", result_value="5")
        self.expect_expr("derived_with_dtor_but_no_base_dtor_as_base.foo()", result_type="int", result_value="6")

    def test_call_overloaded(self):
        self.common_setup()
        self.expect("expr derived_with_overload.foo()", error=True, substrs=["too few arguments to function call, expected 1, have 0"])
        self.expect_expr("derived_with_overload.foo(1)", result_type="int", result_value="7")
        self.expect_expr("derived_with_overload_and_using.foo(1)", result_type="int", result_value="8")
        # FIXME: It seems the using declaration doesn't import the overload from the base class.
        self.expect("expr derived_with_overload_and_using.foo()", error=True, substrs=["too few arguments to function call, expected 1, have 0"])
