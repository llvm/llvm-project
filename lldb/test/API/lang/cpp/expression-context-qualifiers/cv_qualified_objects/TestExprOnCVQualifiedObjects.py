import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self,
            "Break here",
            lldb.SBFileSpec("main.cpp"),
        )

        self.expect_expr("f.bar()", result_type="double", result_value="5")
        self.expect_expr("cf.bar()", result_type="int", result_value="2")
        self.expect_expr("vf.bar()", result_type="short", result_value="8")
        self.expect_expr(
            "cvf.bar()", result_type="const char *", result_summary='"volatile"'
        )
