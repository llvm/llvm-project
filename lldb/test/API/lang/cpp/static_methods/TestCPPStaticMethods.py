"""
Tests expressions that distinguish between static and non-static methods.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CPPStaticMethodsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_with_run_command(self):
        """Test that static methods are properly distinguished from regular methods"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// Break here", lldb.SBFileSpec("main.cpp"))

        self.expect_expr("A::getStaticValue()", result_type="int", result_value="5")
        self.expect_expr("a.getMemberValue()", result_type="int", result_value="3")
