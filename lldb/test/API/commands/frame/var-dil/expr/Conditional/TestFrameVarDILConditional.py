"""
Test DIL ternary conditional operator.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILConditional(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_conditional(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        # DIL evaluates only the operand chosen by the condition,
        # and doesn't check the type or evaluate the other operand.
        # Check integer values.
        self.expect_var_path("true ? sh : sh", value="2", type="short")
        self.expect_var_path("true ? sh : 1", value="2", type="short")
        self.expect_var_path("true ? sh : 1.0f", value="2", type="short")
        self.expect_var_path("false ? 1 : sh", value="2", type="short")
        self.expect_var_path("1 + false ? 10 : 20", value="10")
        self.expect_var_path("1 + (false ? 10 : 20)", value="21")
        self.expect_var_path("0 ? 1 ? 2 : 3 : 4 ? 5 : 6", value="5")
        self.expect_var_path("0 ? 1 : 2 ? 3 : 4 ? 5 : 6", value="3")
        self.expect_var_path("6 ? iref ? arr3[0] ? 5 : sh : 2 : i", value="2")

        # Check enums.
        self.expect_var_path("false ? b_enum : a_enum", value="kTwoA")
        self.expect_var_path("false ? sh : b_enum", value="kOneB")
        self.expect_var_path("false ? b_enum : sh", value="2")

        # Check references
        self.expect_var_path("iref ? 1 : 2", value="1", type="int")
        self.expect_var_path("true ? iref : 2", value="1", type="int")
        self.expect_var_path("false ? 1 : iref", value="1", type="int")

        # Check pointers and arrays.
        nullptr = "0x" + "00" * self.target().GetAddressByteSize()
        self.expect_var_path("true ? 0 : nullptr", value="0")
        self.expect_var_path("true ? nullptr : 0", value=nullptr)
        self.expect_var_path("true ? arr2 : arr3", type="int[2]")
        self.expect_var_path("true ? arr2 : 0", type="int[2]")
        self.expect_var_path("true ? 0 : arr2", value="0")
        self.expect_var_path("true ? nullptr : arr2", value=nullptr)
        self.expect_var_path("*(true ? arr2 : arr3)", value="1")

        # Check result with incompatible type operands.
        # (these would return an error in C++)
        self.expect_var_path("true ? s : 1", type="S")
        self.expect_var_path("true ? 1 : t", value="1")
        self.expect_var_path("true ? nullptr : 1", value=nullptr)
        self.expect_var_path("true ? 1.25 : arr", value="1.25")
        self.expect_var_path("*(true ? iptr : 2.0)", value="1")
        self.expect_var_path("&(true ? i : arr)", type="int *")
        self.expect_var_path("true ? iptr : nullptr", type="int *")

        # Check non-existent values
        self.expect_var_path("true ? 1 : __doesnt_exist", value="1")
        self.expect_var_path("false ? __doesnt_exist : 2", value="2")

        # Use different types in bool context.
        self.expect_var_path("iptr ? 1 : 2", value="1")
        self.expect_var_path("nptr ? 1 : 2", value="2")
        self.expect_var_path("arr2 ? 1 : 2", value="1")
        self.expect_var_path("1.0f ? 1 : 2", value="1")
        self.expect_var_path("a_enum ? 1 : 2", value="1")
        self.expect(
            "frame var -- 's ? 1 : 2'",
            error=True,
            substrs=["value of type 'S' is not contextually convertible to 'bool'"],
        )
