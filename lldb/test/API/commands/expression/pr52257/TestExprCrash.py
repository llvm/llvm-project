"""
Verify that LLDB doesn't crash during expression evaluation.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCrashTestCase(TestBase):

    def test_pr52257(self):
        self.build()
        self.createTestTarget()
        self.expect_expr("b", result_type="B", result_children=[ValueCheck(name="tag_set_")])
