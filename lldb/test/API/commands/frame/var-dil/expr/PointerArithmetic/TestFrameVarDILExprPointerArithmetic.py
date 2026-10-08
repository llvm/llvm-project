"""
Test DIL pointer arithmetic.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILExprPointerArithmetic(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_pointer_arithmetic(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        self.expect_var_path("+array", type="int *")
        self.expect_var_path("+array_ref", type="int *")
        self.expect_var_path("+p_int0", type="int *")

        # Binary operations
        self.expect_var_path("p_char", type="const char *")
        self.expect_var_path("p_char + 1", type="const char *")
        self.expect_var_path("p_char + offset", type="const char *")
        self.expect_var_path("p_char5 + -1", type="const char *")
        self.expect_var_path("p_char5 - 1", type="const char *")
        self.expect_var_path("p_char5 - offset", type="const char *")

        self.expect_var_path("my_p_char", type="my_char_ptr")
        self.expect_var_path("my_p_char + 1", type="my_char_ptr")
        self.expect_var_path("my_p_char - 1", type="my_char_ptr")

        self.expect_var_path("*(p_char + 0)", value="'h'")
        self.expect_var_path("*(5 + p_char)", value="'!'")
        self.expect_var_path("*(p_char5 + -5)", value="'h'")
        self.expect_var_path("*(p_char5 - 5)", value="'h'")
        self.expect_var_path("*(p_char - -5)", value="'!'")
        self.expect_var_path("*(p_char5 - offset + 5)", value="'!'")
        self.expect_var_path("*((p_char + offset) - 5)", value="'h'")
        self.expect_var_path("*(p_char + (offset - 5))", value="'h'")

        self.expect_var_path("*p_int0", value="0")
        self.expect_var_path("*cp_int5", value="5")
        self.expect_var_path("*(&*(cp_int5 + 1) - 1)", value="5")

        self.expect_var_path("p_int0 - p_int0", value="0", type="__ptrdiff_t")
        self.expect_var_path("cp_int5 - p_int0", value="5", type="__ptrdiff_t")
        self.expect_var_path("cp_int5 - td_int_ptr0", value="5", type="__ptrdiff_t")
        self.expect_var_path("td_int_ptr0 - cp_int5", value="-5", type="__ptrdiff_t")

        # Check arrays
        self.expect_var_path("array + 1", type="int *")
        self.expect_var_path("1 + array", type="int *")
        self.expect_var_path("array_ref + 1", type="int *")
        self.expect_var_path("1 + array_ref", type="int *")
        self.expect_var_path("array - 1", type="int *")
        self.expect_var_path("array_ref - 1", type="int *")
        self.expect_var_path("array - array", value="0", type="__ptrdiff_t")
        self.expect_var_path("array - array_ref", value="0", type="__ptrdiff_t")
        self.expect_var_path("array_ref - array_ref", value="0", type="__ptrdiff_t")

        # Errors
        self.expect(
            "frame var -- '-p_int0'",
            error=True,
            substrs=["invalid argument type 'int *' to unary expression"],
        )
        self.expect(
            "frame var -- 'cp_int5 - p_char'",
            error=True,
            substrs=[
                "'const int *' and 'const char *' are not pointers to compatible types"
            ],
        )
        self.expect(
            "frame var -- 'p_int0 + cp_int5'",
            error=True,
            substrs=[
                "invalid operands to binary expression ('int *' and 'const int *')"
            ],
        )
        self.expect(
            "frame var -- 'p_void + 1'",
            error=True,
            substrs=["arithmetic on a pointer to void"],
        )
        self.expect(
            "frame var -- 'p_void - 1'",
            error=True,
            substrs=["arithmetic on a pointer to void"],
        )
        self.expect(
            "frame var -- 'p_void - p_char'",
            error=True,
            substrs=[
                "'void *' and 'const char *' are not pointers to compatible types"
            ],
        )
        self.expect(
            "frame var -- 'p_void - p_void'",
            error=True,
            substrs=["arithmetic on pointers to void"],
        )
        self.expect(
            "frame var -- 'pp_void0 - p_char'",
            error=True,
            substrs=[
                "'void **' and 'const char *' are not pointers to compatible types"
            ],
        )
        self.expect(
            "frame var -- 'p_int0 - 1.0'",
            error=True,
            substrs=["invalid operands to binary expression ('int *' and 'double')"],
        )
        self.expect(
            "frame var -- '1.0f + p_int0'",
            error=True,
            substrs=["invalid operands to binary expression ('float' and 'int *')"],
        )
        self.expect(
            "frame var -- '1 - array'",
            error=True,
            substrs=["invalid operands to binary expression ('int' and 'int[10]')"],
        )
        self.expect(
            "frame var -- 'array + array'",
            error=True,
            substrs=["invalid operands to binary expression ('int[10]' and 'int[10]')"],
        )
        self.expect(
            "frame var -- 'array + array'",
            error=True,
            substrs=["invalid operands to binary expression ('int[10]' and 'int[10]')"],
        )
        self.expect(
            "frame var -- 'int_null + 1'",
            error=True,
            substrs=["arithmetic on a nullptr is undefined"],
        )
        self.expect(
            "frame var -- 'int_null - 1'",
            error=True,
            substrs=["arithmetic on a nullptr is undefined"],
        )
        self.expect(
            "frame var -- 'p_char + *((int*) 0)'",
            error=True,
            substrs=["could not get the offset: parent is NULL"],
        )
        self.expect(
            "frame var -- 'p_char - *((int*) 0)'",
            error=True,
            substrs=["could not get the offset: parent is NULL"],
        )
