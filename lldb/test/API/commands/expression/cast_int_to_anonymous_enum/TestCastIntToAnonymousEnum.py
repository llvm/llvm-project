"""
Test Expression Parser regression text to ensure that we handle anonymous
enums importing correctly.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCastIntToAnonymousEnum(TestBase):
    def test_cast_int_to_anonymous_enum(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.expect_expr("(flow_e)0", result_type="flow_e", result_value="A")
