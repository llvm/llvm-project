"""
Test DIL literals.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILLiterals(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_literals(self):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        # Check number literals parsing
        self.expect_var_path("1.0", value="1", type="double")
        self.expect_var_path("1.0f", value="1", type="float")
        self.expect_var_path("0x1.2p+3f", value="9", type="float")
        self.expect_var_path("1", value="1", type="int")
        self.expect_var_path("1u", value="1", type="unsigned int")
        self.expect_var_path("0b1l", value="1", type="long")
        self.expect_var_path("01ul", value="1", type="unsigned long")
        self.expect_var_path("01lu", value="1", type="unsigned long")
        self.expect_var_path("0o1ll", value="1", type="long long")
        self.expect_var_path("0x1ULL", value="1", type="unsigned long long")
        self.expect_var_path("0x1llu", value="1", type="unsigned long long")
        self.expect(
            "frame var '1LLL'",
            error=True,
            substrs=["Failed to parse token as numeric-constant"],
        )
        self.expect(
            "frame var '1ullu'",
            error=True,
            substrs=["Failed to parse token as numeric-constant"],
        )

        # Check integer literal type edge cases (dil::Interpreter::PickIntegerType)
        frame = thread.GetFrameAtIndex(0)
        v = frame.GetValueForVariablePath("argc")
        # Creating an SBType from a BasicType still requires any value from the frame
        int_size = v.GetType().GetBasicType(lldb.eBasicTypeInt).GetByteSize()
        long_size = v.GetType().GetBasicType(lldb.eBasicTypeLong).GetByteSize()
        longlong_size = v.GetType().GetBasicType(lldb.eBasicTypeLongLong).GetByteSize()

        longlong_str = "0x" + "F" * longlong_size * 2
        longlong_str = str(int(longlong_str, 16))
        self.assert_literal_type(frame, longlong_str, lldb.eBasicTypeUnsignedLongLong)
        toolong_str = "0x" + "F" * longlong_size * 2 + "F"
        self.expect(
            f"frame var '{toolong_str}'",
            error=True,
            substrs=[
                "integer literal is too large to be represented in any integer type"
            ],
        )

        # These check only apply if adjacent types have different sizes
        if int_size < long_size:
            # For exmaple, 0xFFFFFFFF and 4294967295 will have different types
            # even though the numeric value is the same
            hex_str = "0x" + "F" * int_size * 2
            dec_str = str(int(hex_str, 16))
            self.assert_literal_type(frame, hex_str, lldb.eBasicTypeUnsignedInt)
            self.assert_literal_type(frame, dec_str, lldb.eBasicTypeLong)
            long_str = "0x" + "F" * int_size * 2 + "F"
            ulong_str = long_str + "u"
            self.assert_literal_type(frame, long_str, lldb.eBasicTypeLong)
            self.assert_literal_type(frame, ulong_str, lldb.eBasicTypeUnsignedLong)
        if long_size < longlong_size:
            hex_str = "0x" + "F" * long_size * 2
            dec_str = str(int(hex_str, 16))
            self.assert_literal_type(frame, hex_str, lldb.eBasicTypeUnsignedLong)
            self.assert_literal_type(frame, dec_str, lldb.eBasicTypeLongLong)
            longlong_str = "0x" + "F" * long_size * 2 + "F"
            ulonglong_str = longlong_str + "u"
            self.assert_literal_type(frame, longlong_str, lldb.eBasicTypeLongLong)
            self.assert_literal_type(
                frame, ulonglong_str, lldb.eBasicTypeUnsignedLongLong
            )

    def assert_literal_type(self, frame, literal, expected_type):
        value = frame.GetValueForVariablePath(literal)
        basic_type = value.GetType().GetBasicType()
        self.assertEqual(basic_type, expected_type)
