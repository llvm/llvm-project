"""
Test the warnings that LLDB emits when parsing Objective-C expressions.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @no_debug_info_test
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.m"))

        # Don't warn about not using the result of getters. This is perfectly
        # fine in the expression parser and LLDB shouldn't warn the user about that.
        result = self.frame().EvaluateExpression("m.m; unknown_var_to_cause_an_error")
        self.assertNotIn("getters should not", str(result.GetError()))
