"""
Test that target var with no variables returns a correct error
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestTargetVarNoVars(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_target_var_no_vars(self):
        self.build()
        lldbutil.run_to_name_breakpoint(self, 'main')
        self.expect("target variable", substrs=['no global variables in current compile unit', 'main.c'], error=True)

