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
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        self.do_test()

    def do_test(self):
        target = self.createTestTarget()

        # Now create a breakpoint in main.c at the source matching
        # "Set a breakpoint here"
        breakpoint = target.BreakpointCreateBySourceRegex(
            "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )
        self.assertTrue(
            breakpoint and breakpoint.GetNumLocations() >= 1, VALID_BREAKPOINT
        )

        error = lldb.SBError()
        # This is the launch info.  If you want to launch with arguments or
        # environment variables, add them using SetArguments or
        # SetEnvironmentEntries

        launch_info = target.GetLaunchInfo()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        # Did we hit our breakpoint?
        from lldbsuite.test.lldbutil import get_threads_stopped_at_breakpoint

        threads = get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertEqual(
            len(threads), 1, "There should be a thread stopped at our breakpoint"
        )
       # The hit count for the breakpoint should be 1.
        self.assertEquals(breakpoint.GetHitCount(), 1)

        frame = threads[0].GetFrameAtIndex(0)
        command_result = lldb.SBCommandReturnObject()
        interp = self.dbg.GetCommandInterpreter()

        # TestCStyleCastBUiltins
        #  self.expect("frame variable '(int)1'", IsOk[])
        #  self.expect("frame variable '(long long)1'", IsOk[])
        #  self.expect("frame variable '(unsigned long)1'", IsOk[])
        #  self.expect("frame variable '(long const const)1'", IsOk[])
        #  self.expect("frame variable '(long const long)1'", IsOk[])

        #  self.expect("frame variable '(char*)1'", IsOk[])
        #  self.expect("frame variable '(long long**)1'", IsOk[])
        #  self.expect("frame variable '(const long const long const* const const)1'", IsOk[])

        self.expect("frame variable '(long&*)1'", error=True,
                  substrs=["'type name' declared as a pointer to a reference"
                           " of type 'long &'\n(long&*)1\n      ^"])

        self.expect("frame variable '(long& &)1'", error=True,
                    substrs=["type name declared as a reference to a reference"
                             "\n"
                             "(long& &)1\n"
                             "       ^"])

        self.expect("frame variable '(long 1)1'", error=True,
                    substrs=["<expr>:1:7: expected 'r_paren', got: <'1' "
                             "(numeric_constant)>\n"
                             "(long 1)1\n"
                             "      ^"])

        # TestCStyleCastBasicType

        # Test with integer literals.
        self.expect("frame variable '(char)1'", substrs=["'\\x01'"])
        self.expect("frame variable '(unsigned char)-1'", substrs=["'\\xff'"])
        self.expect("frame variable '(short)-1'", substrs=["-1"])
        self.expect("frame variable '(unsigned short)-1'", substrs=["65535"])
        self.expect("frame variable '(long long)1'", substrs=["1"])
        self.expect("frame variable '(unsigned long long)-1'",
                    substrs=["18446744073709551615"])
        self.expect("frame variable '(short)65534'", substrs=["-2"])
        self.expect("frame variable '(unsigned short)100000'",
                    substrs=["34464"])
        self.expect("frame variable '(int)false'", substrs=["0"])
        self.expect("frame variable '(int)true'", substrs=["1"])
        self.expect("frame variable '(float)1'", substrs=["1"])
        self.expect("frame variable '(float)1.1'", substrs=["1.10000002"])
        self.expect("frame variable '(float)1.1f'", substrs=["1.10000002"])
        self.expect("frame variable '(float)-1.1'", substrs=["-1.10000002"])
        self.expect("frame variable '(float)-1.1f'", substrs=["-1.10000002"])
        self.expect("frame variable '(float)false'", substrs=["0"])
        self.expect("frame variable '(float)true'", substrs=["1"])
        self.expect("frame variable '(double)1'", substrs=["1"])
        self.expect("frame variable '(double)1.1'",
                    substrs=["1.1000000000000001"])
        self.expect("frame variable '(double)1.1f'",
                    substrs=["1.1000000238418579"])
        self.expect("frame variable '(double)-1.1'",
                    substrs=["-1.1000000000000001"])
        self.expect("frame variable '(double)-1.1f'",
                    substrs=["-1.1000000238418579"])
        self.expect("frame variable '(double)false'", substrs=["0"])
        self.expect("frame variable '(double)true'", substrs=["1"])
        self.expect("frame variable '(int)1.1'", substrs=["1"])
        self.expect("frame variable '(int)1.1f'", substrs=["1"])
        self.expect("frame variable '(int)-1.1'", substrs=["-1"])
        self.expect("frame variable '(long)1.1'", substrs=["1"])
        self.expect("frame variable '(long)-1.1f'", substrs=["-1"])
        self.expect("frame variable '(bool)0'", substrs=["false"])
        self.expect("frame variable '(bool)0.0'", substrs=["false"])
        self.expect("frame variable '(bool)0.0f'", substrs=["false"])
        self.expect("frame variable '(bool)3'", substrs=["true"])
        self.expect("frame variable '(bool)-3'", substrs=["true"])
        self.expect("frame variable '(bool)-3.4'", substrs=["true"])
        self.expect("frame variable '(bool)-0.1'", substrs=["true"])
        self.expect("frame variable '(bool)-0.1f'", substrs=["true"])

        self.expect("frame variable '&(int)1'", error=True,
                    substrs=["cannot take the address of an rvalue of type"
                             " 'int'"])

        # Test with variables.
        self.expect("frame variable '(char)a'", substrs=["'\\x01'"])
        self.expect("frame variable '(unsigned char)na'", substrs=["'\\xff'"])
        self.expect("frame variable '(short)na'", substrs=["-1"])
        self.expect("frame variable '(unsigned short)-a'", substrs=["65535"])
        self.expect("frame variable '(long long)a'", substrs=["1"])
        self.expect("frame variable '(unsigned long long)-1'",
                    substrs=["18446744073709551615"])
        self.expect("frame variable '(float)a'", substrs=["1"])
        self.expect("frame variable '(float)f'", substrs=["1.10000002"])
        self.expect("frame variable '(double)f'",
                    substrs=["1.1000000238418579"])
        self.expect("frame variable '(int)f'", substrs=["1"])
        self.expect("frame variable '(long)f'", substrs=["1"])
        self.expect("frame variable '(bool)finf'", substrs=["true"])
        self.expect("frame variable '(bool)fnan'", substrs=["true"])
        self.expect("frame variable '(bool)fsnan'", substrs=["true"])
        self.expect("frame variable '(bool)fmax'", substrs=["true"])
        self.expect("frame variable '(bool)fdenorm'", substrs=["true"])
        self.expect("frame variable '(int)ns_foo_'", error=True,
                    substrs=["cannot convert 'ns::Foo' to 'int' without a "
                             "conversion operator"])

        # Test with typedefs and namespaces.
        self.expect("frame variable '(myint)1'", substrs=["1"])
        self.expect("frame variable '(myint)1LL'", substrs=["1"])
        self.expect("frame variable '(ns::myint)1'", substrs=["1"])
        self.expect("frame variable '(::ns::myint)1'", substrs=["1"])
        self.expect("frame variable '(::ns::myint)myint_'", substrs=["1"])

        self.expect("frame variable '(int)myint_'", substrs=["1"])
        self.expect("frame variable '(int)ns_myint_'", substrs=["2"])
        self.expect("frame variable '(long long)myint_'", substrs=["1"])
        self.expect("frame variable '(long long)ns_myint_'", substrs=["2"])
        self.expect("frame variable '(::ns::myint)myint_'", substrs=["1"])

        self.expect("frame variable '(ns::inner::mydouble)1'", substrs=["1"])
        self.expect("frame variable '(::ns::inner::mydouble)1.2'",
                    substrs=["1.2"])
        self.expect("frame variable '(ns::inner::mydouble)myint_'",
                    substrs=["1"])
        self.expect("frame var '(::ns::inner::mydouble)ns_inner_mydouble_'",
                    substrs=["1.2"])
        self.expect("frame variable '(myint)ns_inner_mydouble_'",
                    substrs=["1"])

        # Test with pointers and arrays.
        #  self.expect("frame variable '(long long)ap'", IsOk[])
        #  self.expect("frame variable '(unsigned long long)vp'", IsOk[])
        #  self.expect("frame variable '(long long)arr'", IsOk[])
        self.expect("frame variable '(bool)ap'", substrs=["true"])
        self.expect("frame variable '(bool)(int*)0x00000000'",
                    substrs=["false"])
        self.expect("frame variable '(bool)nullptr'", substrs=["false"])
        self.expect("frame variable '(bool)arr'", substrs=["true"])
        self.expect("frame variable '(char)ap'", error=True,
                    substrs=["cast from pointer to smaller type 'char' loses "
                             "information"])
        Is32Bit = False
        if process.GetAddressByteSize() == 4:
          Is32Bit = True;

        if Is32Bit:
          pass
        #    self.expect("frame variable '(int)arr'", IsOk[])
        else:
          self.expect("frame variable '(int)arr'", error=True,
                      substrs=["cast from pointer to smaller type 'int' loses"
                               " information"])

        self.expect("frame variable '(float)ap'", error=True,
                    substrs=["C-style cast from 'int *' to 'float' is not "
                             "allowed"])
        self.expect("frame variable '(float)arr'", error=True,
                    substrs=["C-style cast from 'int *' to 'float' is not "
                             "allowed"])

        # TestCStyleCastPointer
        # self.expect("frame variable '(void*)&a'", IsOk[])
        #  self.expect("frame variable '(void*)ap'", IsOk[])
        #  self.expect("frame variable '(long long*)vp'", IsOk[])
        #  self.expect("frame variable '(short int*)vp'", IsOk[])
        #  self.expect("frame variable '(unsigned long long*)vp'", IsOk[])
        #  self.expect("frame variable '(unsigned short int*)vp'", IsOk[])


        if Is32Bit:
          self.expect("frame variable '(void*)0'",
                      substrs=["0x00000000"])
          self.expect("frame variable '(void*)1'",
                      substrs=["0x00000001"])
          self.expect("frame variable '(void*)a'",
                      substrs=["0x00000001"])
          self.expect("frame variable '(void*)na'",
                      substrs=["0xffffffff"])
        else:
          self.expect("frame variable '(void*)0'",
                      substrs=["0x0000000000000000"])
          self.expect("frame variable '(void*)1'",
                      substrs=["0x0000000000000001"])
          self.expect("frame variable '(void*)a'",
                      substrs=["0x0000000000000001"])
          self.expect("frame variable '(void*)na'",
                      substrs=["0xffffffffffffffff"])

        #  self.expect("frame variable '(int*&)ap'", IsOk[])

        self.expect("frame variable '(char*) 1.0'", error=True,
                    substrs=["cannot cast from type 'double' to pointer type"
                             " 'char *'"])

        self.expect("frame variable '*(const int* const)ap'", substrs=["1"])
        self.expect("frame variable '*(volatile int* const)ap'", substrs=["1"])
        self.expect("frame variable '*(const int* const)vp'", substrs=["1"])
        self.expect("frame variable '*(const int* const volatile const)vp'",
                    substrs=["1"])
        self.expect("frame variable '*(int*)(void*)ap'", substrs=["1"])
        self.expect("frame variable '*(int*)(const void* const volatile)ap'",
                    substrs=["1"])

        #  self.expect("frame variable '(ns::Foo*)ns_inner_foo_ptr_'", IsOk[])
        #  self.expect("frame variable '(ns::inner::Foo*)ns_foo_ptr_'", IsOk[])

        self.expect("frame variable '(int& &)ap'", error=True,
                    substrs=["type name declared as a reference to a "
                             "reference"])
        self.expect("frame variable '(int&*)ap'", error=True,
                    substrs=["'type name' declared as a pointer "
                             "to a reference of type 'int &'"])

        if Is32Bit:
          self.expect("frame variable '(void *)nullptr'",
                      substrs=["0x00000000"])
          self.expect("frame variable '(void *)0'",
                      substrs=["0x00000000"])
        else:
          self.expect("frame variable '(void *)nullptr'",
                      substrs=["0x0000000000000000"])
          self.expect("frame variable '(void *)0'",
                      substrs=["0x0000000000000000"])

        ##ifndef __EMSCRIPTEN__
        #  self.expect("frame variable '(std::nullptr_t)1'", error=True,
        #              substrs=["C-style cast from 'int' to 'std::nullptr_t' (aka "
        #                      "'nullptr_t') is not allowed"])
        #  self.expect("frame variable '(std::nullptr_t)ap"),
        #              error=True, substrs=["C-style cast from 'int *' to 'std::nullptr_t' (aka "
        #                      "'nullptr_t') is not allowed"])
        ##else
##        self.expect("frame variable '(std::nullptr_t)1'",
##                    error=True, substrs=["C-style cast from 'int' to "
##                                         "'std::nullptr_t' is not allowed"])
##        self.expect("frame variable '(std::nullptr_t)ap'",
##                    error=True, substrs=["C-style cast from 'int *' to "
##                                         "'std::nullptr_t' is not allowed"])
        ##endif


        # TestCStyleCastNullptrType

        if Is32Bit:
          PASS
        #    self.expect("frame variable '(int)nullptr'", IsOk[])
        else:
          self.expect("frame variable '(int)nullptr'", error=True,
                      substrs=["cast from pointer to smaller type 'int' loses"
                               " information"])
        self.expect("frame variable '(uint64_t)nullptr'", substrs=["0"])

        if Is32Bit:
          self.expect("frame variable '(void*)nullptr'",
                      substrs=["0x00000000"])
          self.expect("frame variable '(char*)nullptr'",
                      substrs=["0x00000000"])
        else:
          self.expect("frame variable '(void*)nullptr'",
                      substrs=["0x0000000000000000"])
          self.expect("frame variable '(char*)nullptr'",
                      substrs=["0x0000000000000000"])


        # TestCStyleCastArray

        error = process.Continue();
        threads = get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertEqual(
            len(threads), 1, "There should be a thread stopped at our breakpoint"
        )
       # The hit count for the breakpoint should be 2.
        self.assertEquals(breakpoint.GetHitCount(), 2)

        self.expect("frame variable '(int*)arr_1d'", patterns=["0x[0-9]+"])
        self.expect("frame variable '(char*)arr_1d'", patterns=["0x[0-9]+"])
        self.expect("frame variable '((char*)arr_1d)[0]'", substrs=["'\\x01'"])
        self.expect("frame variable '((char*)arr_1d)[1]'", substrs=["'\\0'"])

        # 2D arrays.
        self.expect("frame variable '(int*)arr_2d'", patterns=["0x[0-9]+"])
        self.expect("frame variable '((int*)arr_2d)[1]'", substrs=["2"])
        self.expect("frame variable '((int*)arr_2d)[2]'", substrs=["3"])
        self.expect("frame variable '((int*)arr_2d[1])[1]'", substrs=["5"])

        # TestCStyleCastReference

        #self.expect("frame variable '((InnerFoo&)arr_1d[1]).a'", substrs=["2"])
        #self.expect("frame variable '((InnerFoo&)arr_1d[1]).b'", substrs=["3"])

        self.expect("frame variable '(int&)arr_1d[0]'", substrs=["1"])
        self.expect("frame variable '(int&)arr_1d[1]'", substrs=["2"])

        self.expect("frame variable '(int&)0'", error=True,
                    substrs=["C-style cast from rvalue to reference type "
                             "'int &'"])
        self.expect("frame variable '&(int&)arr_1d'", patterns=["0x[0-9]+"])
