import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCppChainedCalls(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_with_run_command(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))

        # Test chained calls
        self.expect_expr("get(set(true))", result_type="bool", result_value="true")
        self.expect_expr("get(set(false))", result_type="bool", result_value="false")
        self.expect_expr("get(t & f)", result_type="bool", result_value="false")
        self.expect_expr("get(f & t)", result_type="bool", result_value="false")
        self.expect_expr("get(t & t)", result_type="bool", result_value="true")
        self.expect_expr("get(f & f)", result_type="bool", result_value="false")
        self.expect_expr("get(t & f)", result_type="bool", result_value="false")
        self.expect_expr("get(f) && get(t)", result_type="bool", result_value="false")
        self.expect_expr("get(f) && get(f)", result_type="bool", result_value="false")
        self.expect_expr("get(t) && get(t)", result_type="bool", result_value="true")
