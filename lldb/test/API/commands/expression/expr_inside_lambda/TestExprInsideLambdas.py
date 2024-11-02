""" Test that evaluating expressions from within C++ lambdas works
    Particularly, we test the case of capturing "this" and
    using members of the captured object in expression evaluation
    while we're on a breakpoint inside a lambda.
"""


import lldb
from lldbsuite.test.lldbtest import *


class ExprInsideLambdaTestCase(TestBase):

    def expectExprError(self, expr : str, expected : str):
        frame = self.thread.GetFrameAtIndex(0)
        value = frame.EvaluateExpression(expr)
        errmsg = value.GetError().GetCString()
        self.assertIn(expected, errmsg)

    def test_expr_inside_lambda(self):
        """Test that lldb evaluating expressions inside lambda expressions works correctly."""
        self.build()
        (target, process, self.thread, bkpt) = \
                lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.cpp"))

        # Inside 'Foo::method'

        # Check access to captured 'this'
        self.expect_expr("class_var", result_type="int", result_value="109")
        self.expect_expr("this->class_var", result_type="int", result_value="109")

        # Check that captured shadowed variables take preference over the
        # corresponding member variable
        self.expect_expr("shadowed", result_type="int", result_value="5")
        self.expect_expr("this->shadowed", result_type="int", result_value="-137")

        # Check access to local captures
        self.expect_expr("local_var", result_type="int", result_value="137")
        self.expect_expr("*class_ptr", result_type="int", result_value="137")

        # Check access to base class variables
        self.expect_expr("base_var", result_type="int", result_value="14")
        self.expect_expr("base_base_var", result_type="int", result_value="11")

        # Check access to global variable
        self.expect_expr("global_var", result_type="int", result_value="-5")

        # Check access to multiple captures/member variables
        self.expect_expr("(shadowed + this->shadowed) * (base_base_var + local_var - class_var)",
                         result_type="int", result_value="-5148")

        # Check base-class function call
        self.expect_expr("baz_virt()", result_type="int", result_value="2")
        self.expect_expr("base_var", result_type="int", result_value="14")
        self.expect_expr("this->shadowed", result_type="int", result_value="-1")
        
        # 'p this' should yield 'struct Foo*'
        frame = self.thread.GetFrameAtIndex(0)
        outer_class_addr = frame.GetValueForVariablePath("this->this")
        self.expect_expr("this", result_value=outer_class_addr.GetValue())

        lldbutil.continue_to_breakpoint(process, bkpt)

        # Inside 'nested_lambda'
        
        # Check access to captured 'this'. Should still be 'struct Foo*'
        self.expect_expr("class_var", result_type="int", result_value="109")
        self.expect_expr("global_var", result_type="int", result_value="-5")
        self.expect_expr("this", result_value=outer_class_addr.GetValue())

        # Check access to captures
        self.expect_expr("lambda_local_var", result_type="int", result_value="5")
        self.expect_expr("local_var", result_type="int", result_value="137")

        # Check access to variable in previous frame which we didn't capture
        self.expectExprError("local_var_copy", "use of undeclared identifier")

        lldbutil.continue_to_breakpoint(process, bkpt)

        # By-ref mutates source variable
        self.expect_expr("lambda_local_var", result_type="int", result_value="0")

        # By-value doesn't mutate source variable
        self.expect_expr("local_var_copy", result_type="int", result_value="136")
        self.expect_expr("local_var", result_type="int", result_value="137")

        lldbutil.continue_to_breakpoint(process, bkpt)

        # Inside 'LocalLambdaClass::inner_method'

        # Check access to captured 'this'
        self.expect_expr("lambda_class_local", result_type="int", result_value="-12345")
        self.expect_expr("this->lambda_class_local", result_type="int", result_value="-12345")
        self.expect_expr("outer_ptr->class_var", result_type="int", result_value="109")

        # 'p this' should yield 'struct LocalLambdaClass*'
        frame = self.thread.GetFrameAtIndex(0)
        local_class_addr = frame.GetValueForVariablePath("this->this")
        self.assertNotEqual(local_class_addr, outer_class_addr)
        self.expect_expr("this", result_value=local_class_addr.GetValue())

        # Can still access global variable
        self.expect_expr("global_var", result_type="int", result_value="-5")

        # Check access to outer top-level structure's members
        self.expectExprError("class_var", ("use of non-static data member"
                                           " 'class_var' of 'Foo' from nested type"))

        self.expectExprError("base_var", ("use of non-static data member"
                                           " 'base_var'"))

        self.expectExprError("local_var", ("use of non-static data member 'local_var'"
                                           " of '(unnamed class)' from nested type 'LocalLambdaClass'"))

        # Inside non_capturing_method
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect_expr("local", result_type="int", result_value="5")
        self.expect_expr("local2", result_type="int", result_value="10")
        self.expect_expr("local2 * local", result_type="int", result_value="50")

        self.expectExprError("class_var", ("use of non-static data member"
                                           " 'class_var' of 'Foo' from nested type"))
