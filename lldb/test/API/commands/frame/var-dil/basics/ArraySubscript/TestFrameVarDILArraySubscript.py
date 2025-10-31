"""
Test DIL array subscript.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILArraySubscript(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def expect_var_path(self, expr, compare_to_framevar=False, value=None, type=None):
        value_dil = super().expect_var_path(expr, value=value, type=type)
        if compare_to_framevar:
            self.runCmd("settings set target.experimental.use-DIL false")
            value_frv = super().expect_var_path(expr, value=value, type=type)
            self.runCmd("settings set target.experimental.use-DIL true")
            self.assertEqual(value_dil.GetValue(), value_frv.GetValue())

    def test_subscript(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        # Test int[] and int*
        self.expect_var_path("int_arr[0]", True, value="1")
        self.expect_var_path("int_ptr[1]", True, value="2")
        self.expect("frame var 'int_arr[enum_one]'", error=True)

        # Test when base and index are references.
        self.expect_var_path("int_arr[0]", True, value="1")
        self.expect("frame var 'int_arr[idx_1_ref]'", error=True)
        self.expect("frame var 'int_arr[enum_ref]'", error=True)
        self.expect_var_path("int_arr_ref[0]", value="1")
        self.expect("frame var 'int_arr_ref[idx_1_ref]'", error=True)
        self.expect("frame var 'int_arr_ref[enum_ref]'", error=True)

        # Test when base and index are typedefs.
        self.expect_var_path("td_int_arr[0]", True, value="1")
        self.expect("frame var 'td_int_arr[td_int_idx_1]'", error=True)
        self.expect("frame var 'td_int_arr[td_td_int_idx_2]'", error=True)
        self.expect_var_path("td_int_ptr[0]", True, value="1")
        self.expect("frame var 'td_int_ptr[td_int_idx_1]'", error=True)
        self.expect("frame var 'td_int_ptr[td_td_int_idx_2]'", error=True)

        # Both typedefs and refs
        self.expect("frame var 'td_int_arr_ref[td_int_idx_1_ref]'", error=True)

        # Test for index out of bounds. 1 beyond the end.
        self.expect_var_path("int_arr[3]", True, type="int")
        # Far beyond the end (but not far enough to be off the top of the stack).
        self.expect_var_path("int_arr[10]", True, type="int")

        # Test address-of of the subscripted value.
        self.expect_var_path("*(&int_arr[1])", value="2")

        # Test for negative index.
        self.expect_var_path("int_ptr_1[-1]", True, value="1")

        # Test for floating point index
        self.expect(
            "frame var 'int_arr[1.0]'",
            error=True,
            substrs=["failed to parse integer constant: <'1.0' (float_constant)>"],
        )

        # Test accessing bits in scalar types.
        self.expect_var_path("idx_1[0]", value="1")
        self.expect_var_path("idx_1[1]", value="0")
        self.expect_var_path("1[0]", value="1")

        # Bit access not valid for a reference.
        self.expect(
            "frame var 'idx_1_ref[0]'",
            error=True,
            substrs=["bitfield range 0-0 is not valid"],
        )

        # Base should be a "pointer to T" and index should be of an integral type.
        self.expect(
            "frame var 'int_arr[int_ptr]'",
            error=True,
            substrs=["failed to parse integer constant"],
        )

        # Base should not be a pointer to void
        self.expect(
            "frame var 'p_void[0]'",
            error=True,
            substrs=["subscript of pointer to incomplete type 'void'"],
        )

    def test_subscript_synthetic(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")
        self.runCmd("script from myArraySynthProvider import *")
        self.runCmd("type synth add -l myArraySynthProvider myArray")

        # Test synthetic value subscription
        self.expect_var_path("vector[1]", value="2")
        self.expect(
            "frame var 'vector[100]'",
            error=True,
            substrs=["array index 100 is not valid"],
        )
        self.expect(
            "frame var 'ma_ptr[0]'",
            substrs=["(myArray) ma_ptr[0] = ([0] = 7, [1] = 8, [2] = 9, [3] = 10)"],
        )
