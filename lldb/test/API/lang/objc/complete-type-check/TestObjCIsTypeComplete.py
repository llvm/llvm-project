""" Tests SBType.IsTypeComplete on Objective-C types. """

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @skipUnlessDarwin
    @no_debug_info_test
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m")
        )

        # A class that is only forward declared is not complete.
        incomplete = self.expect_expr("incomplete", result_type="IncompleteClass *")
        self.assertTrue(incomplete.IsValid())
        incomplete_class = incomplete.GetType().GetPointeeType()
        self.assertTrue(incomplete_class.IsValid())
        self.assertFalse(incomplete_class.IsTypeComplete())

        # A class that has its interface fully declared is complete.
        complete = self.expect_expr("complete", result_type="CompleteClass *")
        self.assertTrue(complete.IsValid())
        complete_class = complete.GetType().GetPointeeType()
        self.assertTrue(complete_class.IsValid())
        self.assertTrue(complete_class.IsTypeComplete())

        # A class that has its interface fully declared and an implementation
        # is also complete.
        complete_with_impl = self.expect_expr(
            "complete_impl", result_type="CompleteClassWithImpl *"
        )
        self.assertTrue(complete_with_impl.IsValid())
        complete_class_with_impl = complete_with_impl.GetType().GetPointeeType()
        self.assertTrue(complete_class_with_impl.IsValid())
        self.assertTrue(complete_class_with_impl.IsTypeComplete())
