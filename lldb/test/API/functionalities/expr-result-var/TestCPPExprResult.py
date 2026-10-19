"""
Test the reuse of  C++ result variables, particularly making sure
that the dynamic typing is preserved.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfWasm  # no expression evaluation
class TestCPPResultVariables(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    def setUp(self):
        TestBase.setUp(self)
        self.main_source_file = lldb.SBFileSpec("two-bases.cpp")

    def check_dereference(self, result_varname, frame, expr_options):
        # All the variables we are comparing against are various ways to get
        # pointers to my_derived.  So use that to make our CheckValue:
        my_derived = frame.FindVariable("my_derived")
        self.assertSuccess(my_derived.error, "Got my_derived")
        my_value_check = ValueCheck(valobj=my_derived)

        direct_access_expr = "{0}->derived_int".format(result_varname)
        self.expect_expr(direct_access_expr, result_type="int", result_value="1000")

        # Also check this by directly accessing the result variable:
        result_value = frame.FindValue(result_varname, lldb.eValueTypeConstResult, True)
        self.assertTrue(result_value.error.success, "Found my result variable")
        my_value_check.check_value(
            self, result_value.Dereference(), "children are correct"
        )

        # Make sure we can also call a function through the derived type:
        method_result = self.expect_expr(
            f"{result_varname}->method_of_derived()",
            result_type="int",
            options=expr_options,
        )
        self.assertEqual(method_result.signed, 500, "Got the right result value")

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24663")
    def test_virtual_dynamic_results(self):
        self.do_test_dynamic_results(True)

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24663")
    def test_non_virtual_dynamic_results(self):
        self.do_test_dynamic_results(False)

    def do_test_dynamic_results(self, virtual):
        """Test that when we uses a result variable in a subsequent expression it
        uses the dynamic value - if that was requested when the result variable was made.
        """
        if virtual:
            self.build(dictionary={"CFLAGS_EXTRAS": "-DVIRTUAL=''"})
        else:
            self.build(dictionary={"CFLAGS_EXTRAS": "-DVIRTUAL='virtual'"})

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source_file
        )

        frame = thread.GetFrameAtIndex(0)
        expr_options = lldb.SBExpressionOptions()
        expr_options.SetFetchDynamicValue(lldb.eDynamicDontRunTarget)
        base_1_ptr = self.expect_expr(
            "base_1_ptr", result_type="Derived *", options=expr_options
        )
        result_varname = base_1_ptr.GetName()
        self.check_dereference(result_varname, frame, expr_options)

        # Now do the same thing, but use a persistent result variable:
        empty_var = frame.EvaluateExpression(
            "void *$base_1_ptr = base_1_ptr", expr_options
        )
        self.assertIn(
            empty_var.error.description,
            "unknown error",
            "Expressions that don't have results return this error",
        )
        persist_base_1_ptr = frame.FindValue(
            "$base_1_ptr", lldb.eValueTypeConstResult, True
        )
        self.assertTrue(persist_base_1_ptr.error.success, "Got the persistent variable")
        self.check_dereference("$base_1_ptr", frame, expr_options)

        # Now check the second of the multiply inherited bases, this one will have an offset_to_top
        # that we need to calculate:
        base_2_ptr = self.expect_expr(
            "base_2_ptr", result_type="Derived *", options=expr_options
        )
        self.check_dereference(base_2_ptr.GetName(), frame, expr_options)

        # Again, do the same thing for a persistent expression variable:
        empty_var = frame.EvaluateExpression(
            "void *$base_2_ptr = base_2_ptr", expr_options
        )
        self.check_dereference("$base_2_ptr", frame, expr_options)

        # Now try starting from a virtual base class of both our bases:
        base_through_1 = self.expect_expr(
            "base_through_1", result_type="Derived *", options=expr_options
        )
        self.check_dereference(base_through_1.GetName(), frame, expr_options)

        # Now try starting from a virtual base class of both our bases:
        base_through_2 = self.expect_expr(
            "base_through_2", result_type="Derived *", options=expr_options
        )
        self.check_dereference(base_through_2.GetName(), frame, expr_options)

        # Now check that we get the right results when we run an
        # expression to get the base class object:
        base_through_expr = self.expect_expr(
            "MakeADerivedReportABase()", result_type="Derived *", options=expr_options
        )
        self.check_dereference(base_through_expr.GetName(), frame, expr_options)
