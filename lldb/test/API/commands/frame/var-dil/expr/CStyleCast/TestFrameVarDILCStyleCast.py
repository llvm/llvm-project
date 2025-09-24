"""
Make sure 'frame var' using DIL parser/evaultor works for C-Style casts.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil

import os
import shutil
import time


class TestFrameVarDILCStyleCast(TestBase):
    def test_type_cast(self):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        # TestCStyleCastBUiltins

        self.expect_var_path("(int)1", value="1", type="int")
        self.expect_var_path("(long long)1", value="1", type="long long")
        self.expect_var_path("(unsigned long)1", value="1", type="unsigned long")
        self.expect_var_path("(char*)1", value="0x0000000000000001", type="char *")
        self.expect_var_path(
            "(long long**)1", value="0x0000000000000001", type="long long **"
        )

        self.expect(
            "frame variable '(long&*)1'",
            error=True,
            substrs=[
                "'type name' declared as a pointer to a reference of type 'long &'"
            ],
        )

        self.expect(
            "frame variable '(long& &)1'",
            error=True,
            substrs=["type name declared as a reference to a reference"],
        )

        self.expect(
            "frame variable '(long 1)1'",
            error=True,
            substrs=["expected 'r_paren', got: <'1' (integer_constant)>"],
        )

        # TestCStyleCastBasicType

        # Test with integer literals.
        self.expect_var_path("(char)1", type="char", value="'\\x01'")
        self.expect_var_path("(long long)1", type="long long", value="1")
        self.expect_var_path("(short)65534", type="short", value="-2")
        self.expect_var_path(
            "(unsigned short)100000", type="unsigned short", value="34464"
        )
        self.expect_var_path("(int)false", type="int", value="0")
        self.expect_var_path("(int)true", type="int", value="1")
        self.expect_var_path("(float)1", type="float", value="1")
        self.expect_var_path("(float)1.1", type="float", value="1.10000002")
        self.expect_var_path("(float)1.1f", type="float", value="1.10000002")
        self.expect_var_path("(float)false", type="float", value="0")
        self.expect_var_path("(float)true", type="float", value="1")
        self.expect_var_path("(double)1", type="double", value="1")
        self.expect_var_path("(double)1.1", type="double", value="1.1000000000000001")
        self.expect_var_path("(double)1.1f", type="double", value="1.1000000238418579")
        self.expect_var_path("(double)false", type="double", value="0")
        self.expect_var_path("(double)true", type="double", value="1")
        self.expect_var_path("(int)1.1", type="int", value="1")
        self.expect_var_path("(int)1.1f", type="int", value="1")
        self.expect_var_path("(long)1.1", type="long", value="1")
        self.expect_var_path("(bool)0", type="bool", value="false")
        self.expect_var_path("(bool)0.0", type="bool", value="false")
        self.expect_var_path("(bool)0.0f", type="bool", value="false")
        self.expect_var_path("(bool)3", type="bool", value="true")

        self.expect(
            "frame variable '&(int)1'",
            error=True,
            substrs=["'result' doesn't have a valid address"],
        )

        # Test with variables.
        self.expect_var_path("(char)a", type="char", value="'\\x01'")
        self.expect_var_path("(unsigned char)na", type="unsigned char", value="'\\xff'")
        self.expect_var_path("(short)na", type="short", value="-1")
        self.expect_var_path("(long long)a", type="long long", value="1")
        self.expect_var_path("(float)a", type="float", value="1")
        self.expect_var_path("(float)f", type="float", value="1.10000002")
        self.expect_var_path("(double)f", type="double", value="1.1000000238418579")
        self.expect_var_path("(int)f", type="int", value="1")
        self.expect_var_path("(long)f", type="long", value="1")
        self.expect_var_path("(bool)finf", type="bool", value="true")
        self.expect_var_path("(bool)fnan", type="bool", value="true")
        self.expect_var_path("(bool)fsnan", type="bool", value="true")
        self.expect_var_path("(bool)fmax", type="bool", value="true")
        self.expect_var_path("(bool)fdenorm", type="bool", value="true")
        self.expect(
            "frame variable '(int)ns_foo_'",
            error=True,
            substrs=["cannot convert 'ns::Foo' to 'int' without a conversion operator"],
        )

        # Test with typedefs and namespaces.
        self.expect_var_path("(myint)1", type="myint", value="1")
        self.expect_var_path("(myint)1LL", type="myint", value="1")
        self.expect_var_path("(ns::myint)1", type="ns::myint", value="1")
        self.expect_var_path("(::ns::myint)1", type="ns::myint", value="1")
        self.expect_var_path("(::ns::myint)myint_", type="ns::myint", value="1")

        self.expect_var_path("(int)myint_", type="int", value="1")
        self.expect_var_path("(int)ns_myint_", type="int", value="2")
        self.expect_var_path("(long long)myint_", type="long long", value="1")
        self.expect_var_path("(long long)ns_myint_", type="long long", value="2")
        self.expect_var_path("(::ns::myint)myint_", type="ns::myint", value="1")

        self.expect_var_path(
            "(ns::inner::mydouble)1", type="ns::inner::mydouble", value="1"
        )
        self.expect_var_path(
            "(::ns::inner::mydouble)1.2", type="ns::inner::mydouble", value="1.2"
        )
        self.expect_var_path(
            "(ns::inner::mydouble)myint_", type="ns::inner::mydouble", value="1"
        )
        self.expect_var_path(
            "(::ns::inner::mydouble)ns_inner_mydouble_",
            type="ns::inner::mydouble",
            value="1.2",
        )
        self.expect_var_path("(myint)ns_inner_mydouble_", type="myint", value="1")

        # Test with pointers and arrays.
        self.expect_var_path("(long long)ap", type="long long")
        self.expect_var_path("(unsigned long long)vp", type="unsigned long long")
        self.expect_var_path("(long long)arr", type="long long")
        self.expect_var_path("(bool)ap", type="bool", value="true")
        self.expect_var_path("(bool)(int*)0x00000000", type="bool", value="false")
        self.expect_var_path("(bool)arr", type="bool", value="true")
        self.expect(
            "frame variable '(char)ap'",
            error=True,
            substrs=["cast from pointer to smaller type 'char' loses information"],
        )
        Is32Bit = False
        if self.target().GetAddressByteSize() == 4:
            Is32Bit = True

        if Is32Bit:
            self.expect("frame variable '(int)arr'", type="int")
        else:
            self.expect(
                "frame variable '(int)arr'",
                error=True,
                substrs=["cast from pointer to smaller type 'int' loses information"],
            )

        self.expect(
            "frame variable '(float)ap'",
            error=True,
            substrs=["C-style cast from 'int *' to 'float' is not allowed"],
        )
        self.expect(
            "frame variable '(float)arr'",
            error=True,
            substrs=["C-style cast from 'int *' to 'float' is not allowed"],
        )

        # TestCStyleCastPointer
        self.expect_var_path("(void*)&a", type="void *")
        self.expect_var_path("(void*)ap", type="void *")
        self.expect_var_path("(long long*)vp", type="long long *")
        self.expect_var_path("(short int*)vp", type="short *")
        self.expect_var_path("(unsigned long long*)vp", type="unsigned long long *")
        self.expect_var_path("(unsigned short int*)vp", type="unsigned short *")

        if Is32Bit:
            self.expect_var_path("(void*)0", type="void *", value="0x00000000")
            self.expect_var_path("(void*)1", type="void *", value="0x00000001")
            self.expect_var_path("(void*)a", type="void *", value="0x00000001")
            self.expect_var_path("(void*)na", type="void *", value="0xffffffff")
        else:
            self.expect_var_path("(void*)0", type="void *", value="0x0000000000000000")
            self.expect_var_path("(void*)1", type="void *", value="0x0000000000000001")
            self.expect_var_path("(void*)a", type="void *", value="0x0000000000000001")
            self.expect_var_path("(void*)na", type="void *", value="0xffffffffffffffff")

        self.expect_var_path("(int*&)ap", type="int *")

        self.expect(
            "frame variable '(char*) 1.0'",
            error=True,
            substrs=["cannot cast from type 'double' to pointer type 'char *'"],
        )

        self.expect_var_path("*(int*)(void*)ap", type="int", value="1")

        self.expect_var_path("(ns::Foo*)ns_inner_foo_ptr_", type="ns::Foo *")
        self.expect_var_path("(ns::inner::Foo*)ns_foo_ptr_", type="ns::inner::Foo *")

        self.expect(
            "frame variable '(int& &)ap'",
            error=True,
            substrs=["type name declared as a reference to a reference"],
        )
        self.expect(
            "frame variable '(int&*)ap'",
            error=True,
            substrs=[
                "'type name' declared as a pointer to a reference of type 'int &'"
            ],
        )

        if Is32Bit:
            self.expect_var_path("(void *)0", type="void *", value="0x00000000")
        else:
            self.expect_var_path("(void *)0", type="void *", value="0x0000000000000000")

        # TestCStyleCastArray

        self.expect_var_path("(int*)arr_1d", type="int *")
        self.expect_var_path("(char*)arr_1d", type="char *")
        self.expect_var_path("((char*)arr_1d)[0]", type="char", value="'\\x01'")
        self.expect_var_path("((char*)arr_1d)[1]", type="char", value="'\\0'")

        # 2D arrays.
        self.expect_var_path("(int*)arr_2d", type="int *")
        self.expect_var_path("((int*)arr_2d)[1]", type="int", value="2")
        self.expect_var_path("((int*)arr_2d)[2]", type="int", value="3")
        self.expect_var_path("((int*)arr_2d[1])[1]", type="int", value="5")

        # TestCStyleCastReference

        self.expect_var_path("((InnerFoo&)arr_1d[1]).a", type="int", value="2")
        self.expect_var_path("((InnerFoo&)arr_1d[1]).b", type="int", value="3")

        self.expect_var_path("(int&)arr_1d[0]", type="int", value="1")
        self.expect_var_path("(int&)arr_1d[1]", type="int", value="2")

        self.expect_var_path("&(int&)arr_1d", type="int *")
