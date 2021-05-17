import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # On Windows we can lookup the declarations of static members but finding
    # up the underlying symbols doesn't work yet.
    @expectedFailureAll(oslist=["windows"])
    @no_debug_info_test
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break in static member function", lldb.SBFileSpec("main.cpp"))

        # Evaluate a static member and call a static member function.
        self.expect_expr("static_member_var", result_type="int", result_value="2")
        self.expect_expr("static_const_member_var", result_type="const int", result_value="3")
        self.expect_expr("static_constexpr_member_var", result_type="const int", result_value="4")
        self.expect_expr("static_func()", result_type="int", result_value="6")

        # Check that accessing non-static members just reports a diagnostic.
        self.expect("expr member_var", error=True,
                    substrs=["invalid use of member 'member_var' in static member function"])
        self.expect("expr member_func()", error=True,
                    substrs=["call to non-static member function without an object argument"])
        self.expect("expr this", error=True,
                    substrs=["invalid use of 'this' outside of a non-static member function"])

        # Continue to a non-static member function of the same class and make
        # sure that evaluating non-static members now works.
        breakpoint = self.target().BreakpointCreateBySourceRegex(
            "// break in member function", lldb.SBFileSpec("main.cpp"))
        self.assertNotEqual(breakpoint.GetNumResolvedLocations(), 0)
        stopped_threads = lldbutil.continue_to_breakpoint(self.process(), breakpoint)

        self.expect_expr("member_var", result_type="int", result_value="1")
        self.expect_expr("member_func()", result_type="int", result_value="5")
