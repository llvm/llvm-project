"""
Test DIL logical operators.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarLogical(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_logical(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        self.expect_var_path("1 && 2", value="true")
        self.expect_var_path("0 && 1", value="false")
        self.expect_var_path("0 || 1", value="true")
        self.expect_var_path("0 || 0", value="false")

        self.expect_var_path("!1", value="false")
        self.expect_var_path("!!1", value="true")

        self.expect_var_path("!trueVar", value="false")
        self.expect_var_path("!!trueVar", value="true")
        self.expect_var_path("!falseVar", value="true")
        self.expect_var_path("!!falseVar", value="false")
        self.expect_var_path("!trueRef", value="false")
        self.expect_var_path("!!trueRef", value="true")

        self.expect_var_path("trueVar || true", value="true")
        self.expect_var_path("trueVar && false", value="false")
        self.expect_var_path("falseVar || true", value="true")
        self.expect_var_path("falseVar && true", value="false")
        self.expect_var_path("true || false && false", value="true")
        self.expect_var_path("(true || false) && false", value="false")

        self.expect_var_path("!p_ptr", value="false")
        self.expect_var_path("!!p_ptr", value="true")
        self.expect_var_path("p_ptr && true", value="true")
        self.expect_var_path("p_ptr && false", value="false")
        self.expect_var_path("!p_nullptr", value="true")
        self.expect_var_path("!!p_nullptr", value="false")
        self.expect_var_path("p_nullptr || true", value="true")
        self.expect_var_path("p_nullptr || false", value="false")
        self.expect_var_path("nullptr || true", value="true")
        self.expect_var_path("nullptr || false", value="false")

        self.expect_var_path("!array", value="false")
        self.expect_var_path("!!array", value="true")
        self.expect_var_path("array || true", value="true")
        self.expect_var_path("false || array", value="true")
        self.expect_var_path("array && true", value="true")
        self.expect_var_path("array && false", value="false")

        # DIL doesn't evaluate the right operand if the left one
        # determines the result
        self.expect_var_path("true || __doesnt_exist", value="true")
        self.expect_var_path("false && __doesnt_exist", value="false")

        # Check errors
        self.expect(
            "frame var -- 'false || !s'",
            error=True,
            substrs=["invalid argument type 'S' to unary expression"],
        )
        self.expect(
            "frame var -- 's || false'",
            error=True,
            substrs=["value of type 'S' is not contextually convertible to 'bool'"],
        )
        self.expect(
            "frame var -- 'true && s'",
            error=True,
            substrs=["value of type 'S' is not contextually convertible to 'bool'"],
        )
