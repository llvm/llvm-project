"""
Test DIL comparison operators.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarComparison(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_comparison(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        # Check arithmetic comparison
        self.expect_var_path("1 == 1", value="true")
        self.expect_var_path("1 == 1.0", value="true")
        self.expect_var_path("i == 1", value="true")
        self.expect_var_path("iref == i", value="true")
        self.expect_var_path("array[0] == i", value="true")
        self.expect_var_path("trueVar == true", value="true")
        self.expect_var_path("1 == true", value="true")
        self.expect_var_path("array[0] != array[1]", value="true")
        self.expect_var_path("1 != 2 == true", value="true")
        self.expect_var_path("true != 2 < 3", value="false")
        self.expect_var_path("ScopedEnum::kZeroS < ScopedEnum::kOneS", value="true")
        self.expect_var_path("1 > 2", value="false")
        self.expect_var_path("1 > 0.1", value="true")
        self.expect_var_path("1 >= 2", value="false")
        self.expect_var_path("2 >= 2", value="true")
        self.expect_var_path("1.0 <= 1.25", value="true")
        self.expect_var_path("1.25f <= 1.0", value="false")
        self.expect_var_path("1.0 <= 1.0", value="true")

        self.expect(
            "frame var -- 'ScopedEnum::kZeroS < ScopedEnumInt8::kOneS8'",
            error=True,
            substrs=[
                "invalid operands to binary expression "
                "('ScopedEnum' and 'ScopedEnumInt8')"
            ],
        )

        self.expect(
            "frame var -- 's < 4",
            error=True,
            substrs=["invalid operands to binary expression ('S' and 'int')"],
        )

        # Check pointer comparison
        self.expect_var_path("p_void == p_void", value="true")
        self.expect_var_path("p_void == p_char1", value="true")
        self.expect_var_path("p_void != p_char1", value="false")
        self.expect_var_path("p_void > p_char1", value="false")
        self.expect_var_path("p_void >= p_char1", value="true")
        self.expect_var_path("p_void < (p_char1 + 1)", value="true")
        self.expect_var_path("pp_void0 + 1 == pp_void1", value="true")

        self.expect_var_path("(void*)1 == (void*)1", value="true")
        self.expect_var_path("(void*)1 != (void*)1", value="false")
        self.expect_var_path("(void*)2 > (void*)1", value="true")
        self.expect_var_path("(void*)2 < (void*)1", value="false")

        self.expect_var_path("(void*)1 == (char*)1", value="true")
        self.expect_var_path("(char*)1 != (void*)1", value="false")
        self.expect_var_path("(void*)2 > (char*)1", value="true")
        self.expect_var_path("(char*)2 < (void*)1", value="false")

        self.expect_var_path("(void*)0 == 0", value="true")
        self.expect_var_path("0 != (void*)0", value="false")

        self.expect_var_path("(void*)0 == nullptr", value="true")
        self.expect_var_path("(void*)0 != nullptr", value="false")
        self.expect_var_path("nullptr == (void*)1", value="false")
        self.expect_var_path("nullptr != (void*)1", value="true")

        self.expect_var_path("nullptr == nullptr", value="true")
        self.expect_var_path("nullptr != nullptr", value="false")

        self.expect_var_path("nullptr == 0", value="true")
        self.expect_var_path("0 != nullptr", value="false")
        self.expect_var_path("nullptr == 0U", value="true")
        self.expect_var_path("0L != nullptr", value="false")
        self.expect_var_path("nullptr == 0UL", value="true")
        self.expect_var_path("0ULL != nullptr", value="false")
        self.expect_var_path("nullptr == 0x0", value="true")
        self.expect_var_path("0b0 != nullptr", value="false")
        self.expect_var_path("nullptr == 00", value="true")
        self.expect_var_path("0x0LLU != nullptr", value="false")

        self.expect_var_path("0 == std_nullptr_t", value="true")
        self.expect_var_path("std_nullptr_t != 0", value="false")

        self.expect_var_path("array == p_int0", value="true")
        self.expect_var_path("p_int0 == array", value="true")
        self.expect_var_path("array < p_int1", value="true")
        self.expect_var_path("array == nullptr", value="false")

        # These are not allowed by C++, but DIL supports these for convenience.
        self.expect_var_path("(void*)1 == 1", value="true")
        self.expect_var_path("(void*)1 == 0", value="false")
        self.expect_var_path("(void*)1 > 0", value="true")
        self.expect_var_path("(void*)1 < 0", value="false")
        self.expect_var_path("1 > (void*)0", value="true")
        self.expect_var_path("2 < (void*)3", value="true")

        # Integer is converted to uintptr_t, so negative numbers because large
        # positive numbers.
        self.expect_var_path("(void*)-1 == -1", value="true")
        self.expect_var_path("(void*)1 > -1", value="false")

        self.expect(
            "frame var -- '(void*)0 > nullptr'",
            error=True,
            substrs=[
                "invalid operands to binary expression ('void *' and 'std::nullptr_t')"
            ],
        )

        self.expect(
            "frame var -- 'nullptr > 0'",
            error=True,
            substrs=[
                "invalid operands to binary expression ('std::nullptr_t' and 'int')"
            ],
        )

        self.expect(
            "frame var -- '1 == nullptr'",
            error=True,
            substrs=[
                "invalid operands to binary expression ('int' and 'std::nullptr_t')"
            ],
        )

        self.expect(
            "frame var -- 'nullptr == (int)0'",
            error=True,
            substrs=[
                "invalid operands to binary expression ('std::nullptr_t' and 'int')"
            ],
        )

        self.expect(
            "frame var -- 'false == nullptr'",
            error=True,
            substrs=[
                "invalid operands to binary expression ('bool' and 'std::nullptr_t')"
            ],
        )

        self.expect(
            "frame var -- 'p_int0 > p_char1'",
            error=True,
            substrs=[
                "comparison of distinct pointer types ('int *' and 'const char *')"
            ],
        )

        self.expect(
            "frame var -- 'pp_void0 == p_char1'",
            error=True,
            substrs=[
                "comparison of distinct pointer types ('void **' and 'const char *')"
            ],
        )
