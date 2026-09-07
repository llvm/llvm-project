"""
Test DIL operator sizeof().
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILExprSizeOf(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_sizeof(self):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        frame = thread.GetFrameAtIndex(0)
        int_size = frame.GetValueForVariablePath("int_size").GetValue()
        short_size = frame.GetValueForVariablePath("short_size").GetValue()
        double_size = frame.GetValueForVariablePath("double_size").GetValue()
        ptr_size = frame.GetValueForVariablePath("ptr_size").GetValue()
        intref_size = frame.GetValueForVariablePath("intref_size").GetValue()
        arr_size = frame.GetValueForVariablePath("arr_size").GetValue()
        foo_size = frame.GetValueForVariablePath("foo_size").GetValue()
        enum_size = frame.GetValueForVariablePath("enum_size").GetValue()
        bitfield_size = frame.GetValueForVariablePath("bitfield_size").GetValue()

        # Check variables
        self.expect_var_path("sizeof(i)", value=int_size, type="__size_t")
        self.expect_var_path("sizeof(sh)", value=short_size)
        self.expect_var_path("sizeof(d)", value=double_size)
        self.expect_var_path("sizeof(ptr)", value=ptr_size)
        self.expect_var_path("sizeof(iref)", value=intref_size)
        self.expect_var_path("sizeof(arr)", value=arr_size)
        self.expect_var_path("sizeof(arr + 1)", value=ptr_size)
        self.expect_var_path("sizeof(arr[0])", value=int_size)
        self.expect_var_path("sizeof(arr2d[1])", value=arr_size)
        self.expect_var_path("sizeof(i + sh + d)", value=double_size)

        # Check types
        self.expect_var_path("sizeof(int)", value=int_size)
        self.expect_var_path("sizeof(int*)", value=ptr_size)
        self.expect_var_path("sizeof(int***)", value=ptr_size)
        self.expect_var_path("sizeof(short)", value=short_size)
        self.expect_var_path("sizeof(double)", value=double_size)
        self.expect_var_path("sizeof(int&)", value=intref_size)
        self.expect_var_path("sizeof(short&)", value=short_size)
        self.expect_var_path("sizeof(char&)", value="1")
        self.expect_var_path("sizeof(short*&)", value=ptr_size)

        # Check struct
        self.expect_var_path("sizeof(foo)", value=foo_size)
        self.expect_var_path("sizeof(&foo)", value=ptr_size)
        self.expect_var_path("sizeof(*foo_ptr)", value=foo_size)
        self.expect_var_path("sizeof(SizeOfFoo)", value=foo_size)
        self.expect_var_path("sizeof(SizeOfFoo&)", value=foo_size)
        self.expect_var_path("sizeof(SizeOfFoo*)", value=ptr_size)
        self.expect_var_path("sizeof(foo.d)", value=double_size)

        # Check enum
        self.expect_var_path("sizeof(enum_one)", value=enum_size)
        self.expect_var_path("sizeof(&enum_one)", value=ptr_size)
        self.expect_var_path("sizeof(UnscopedEnum16::kOne16)", value=enum_size)
        self.expect_var_path("sizeof(UnscopedEnum16)", value=enum_size)
        self.expect_var_path("sizeof(UnscopedEnum16&)", value=enum_size)
        self.expect_var_path("sizeof(UnscopedEnum16*)", value=ptr_size)

        # Check bitfield
        self.expect_var_path("sizeof(bitfield)", value=bitfield_size)
        self.expect_var_path("sizeof(BitFieldStruct)", value=bitfield_size)
        self.expect(
            "frame var -- 'sizeof(bitfield.a)'",
            error=True,
            substrs=["invalid application of 'sizeof' to bit-field"],
        )

        self.expect(
            "frame var -- 'sizeof(bar)'",
            error=True,
            substrs=["use of undeclared identifier 'bar'"],
        )
