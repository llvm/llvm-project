"""
Make sure 'frame var' using DIL parser/evaultor works for C-Style casts..
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil

import os
import shutil
import time

class TestFrameVarDILArithmetic(TestBase):
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


        # TestCxxStaticCast

        # Cast to scalars.
        self.expect("frame variable 'static_cast<int>(1.1)'", substrs=["1"])
        self.expect("frame variable 'static_cast<double>(1)'", substrs=["1"])
        self.expect("frame variable 'static_cast<char>(128)'", substrs=["'\\x80'"])
        self.expect("frame variable 'static_cast<bool>(nullptr)'", substrs=["false"])
        self.expect("frame variable 'static_cast<bool>((int*)0)'", substrs=["false"])
        self.expect("frame variable 'static_cast<bool>(arr)'", substrs=["true"])
        self.expect("frame variable 'static_cast<int>(u_enum)'", substrs=["2"])
        self.expect("frame variable 'static_cast<float>(s_enum)'", substrs=["1"])
        self.expect("frame variable 'static_cast<td_int_t>(5.3)'", substrs=["5"])
        self.expect("frame variable 'static_cast<td_int_t>(4)'", substrs=["4"])
        self.expect("frame variable 'static_cast<int>(td_int)'", substrs=["13"])

        ##ifndef __EMSCRIPTEN__
        #  EXPECT_THAT(
        #      Eval("static_cast<long long>(nullptr)'",
        #      error=True, substrs=["static_cast from 'nullptr_t' to 'long long' is not allowed"])
        ##else
        self.expect("frame variable 'static_cast<long long>(nullptr)'",
                    error=True, substrs=["static_cast from 'std::nullptr_t' to 'long long' is not "
                                         "allowed"])
        ##endif
        self.expect("frame variable 'static_cast<long long>(ptr)'",
                    error=True, substrs=["static_cast from 'int *' to 'long long' is not allowed"])
        self.expect("frame variable 'static_cast<long long>(arr)'",
                    error=True, substrs=["static_cast from 'int *' to 'long long' is not allowed"])
        self.expect("frame variable 'static_cast<int>(parent)'",
                    error=True, substrs=[
                        "cannot convert 'CxxParent' to 'int' without a conversion operator"])
        self.expect("frame variable 'static_cast<long long>(base)'",
                    error=True, substrs=["static_cast from 'CxxBase *' to 'long long' is not allowed"])

        # Cast to enums.
        self.expect("frame variable 'static_cast<UEnum>(0)'", substrs=["kUZero"])
        self.expect("frame variable 'static_cast<UEnum>(s_enum)'", substrs=["kUOne"])
        self.expect("frame variable 'static_cast<UEnum>(2.1)'", substrs=["kUTwo"])
        self.expect("frame variable 'static_cast<SEnum>(true)'", substrs=["kSOne"])
        self.expect("frame variable 'static_cast<SEnum>(0.4f)'", substrs=["kSZero"])
        self.expect("frame variable 'static_cast<SEnum>(td_senum)'", substrs=["kSOne"])
        self.expect("frame variable 'static_cast<td_senum_t>(UEnum::kUOne)'", substrs=["kSOne"])

        ##ifndef __EMSCRIPTEN__
        #  EXPECT_THAT(
        #      Eval("static_cast<UEnum>(nullptr)'",
        #      error=True, substrs=["static_cast from 'nullptr_t' to 'UEnum' is not allowed"])
        ##else
        self.expect("frame variable 'static_cast<UEnum>(nullptr)'",
                    error=True, substrs=["static_cast from 'std::nullptr_t' to 'UEnum' is not allowed"])
        ##endif
        self.expect("frame variable 'static_cast<UEnum>(ptr)'",
                    error=True, substrs=["static_cast from 'int *' to 'UEnum' is not allowed"])
        self.expect("frame variable 'static_cast<SEnum>(parent)'",
                    error=True, substrs=["static_cast from 'CxxParent' to 'SEnum' is not allowed"])

        Is32Bit = False
        if process.GetAddressByteSize() == 4:
          Is32Bit = True;

        # Cast to pointers & cast to nullptr.
        if Is32Bit:
          self.expect("frame variable 'static_cast<char*>(0)'",
                      substrs=["0x00000000"])
          self.expect("frame variable 'static_cast<long*>(nullptr)'",
                      substrs=["0x00000000"])
          self.expect("frame variable 'static_cast<std::nullptr_t>(nullptr)'",
                      substrs=["0x00000000"])
          self.expect("frame variable 'static_cast<void *>(0)'",
                      substrs=["0x00000000"])
          self.expect("frame variable 'static_cast<int*>((void*)4)'",
                      substrs=["0x00000004"])
        else:
          self.expect("frame variable 'static_cast<char*>(0)'",
                      substrs=["0x0000000000000000"])
          self.expect("frame variable 'static_cast<long*>(nullptr)'",
                      substrs=["0x0000000000000000"])
          self.expect("frame variable 'static_cast<std::nullptr_t>(nullptr)'",
                      substrs=["0x0000000000000000"])
          self.expect("frame variable 'static_cast<void *>(0)'",
                      substrs=["0x0000000000000000"])
          self.expect("frame variable 'static_cast<int*>((void*)4)'",
                      substrs=["0x0000000000000004"])

        self.expect("frame variable '*static_cast<int*>(arr)'", substrs=["1"])
        self.expect("frame variable '*static_cast<td_int_ptr_t>(arr)'", substrs=["1"])
        #  self.expect("frame variable 'static_cast<void*>(ptr)'", IsOk[])
        self.expect("frame variable 'static_cast<long*>(ptr)'",
                    error=True, substrs=["static_cast from 'int *' to 'long *' is not allowed"])
        self.expect("frame variable 'static_cast<float*>(arr)'",
                    error=True, substrs=["static_cast from 'int *' to 'float *' is not allowed"])

        # Cast to nullptr.
        ##ifndef __EMSCRIPTEN__
        #  self.expect("frame variable 'static_cast<std::nullptr_t>((int)0)'",
        #              error=True, substrs=["static_cast from 'int' to 'std::nullptr_t' (aka "
        #                      "'nullptr_t') is not allowed"])
        #  self.expect("frame variable 'static_cast<std::nullptr_t>((void*)0)'",
        #              error=True, substrs=["static_cast from 'void *' to 'std::nullptr_t' (aka "
        #                      "'nullptr_t') is not allowed"])
        ##else
        self.expect("frame variable 'static_cast<std::nullptr_t>((int)0)'",
                    error=True, substrs=["static_cast from 'int' to 'std::nullptr_t' is not allowed"])
        self.expect("frame variable 'static_cast<std::nullptr_t>((void*)0)'",
                    error=True, substrs=["static_cast from 'void *' to 'std::nullptr_t' is not allowed"])
        #endif

        # Cast to references.
        self.expect("frame variable 'static_cast<int&>(parent.b)'", substrs=["2"])
        #  self.expect("frame variable '&static_cast<int&>(parent.b)'", IsOk[])
        self.expect("frame variable 'static_cast<int&>(parent.c)'",
                    error=True, substrs=[
                        "static_cast from 'long long' to 'int &' is not implemented yet"])
        self.expect("frame variable 'static_cast<int&>(5)'",
                    error=True, substrs=["static_cast from rvalue of type 'int' to reference type "
                                         "'int &' is not implemented yet"])


        # Invalid expressions.
        self.expect("frame variable 'static_cast<1>(1)'",
                    error=True, substrs=["type name requires a specifier or qualifier"])
        self.expect("frame variable 'static_cast<>(1)'",
                    error=True, substrs=["type name requires a specifier or qualifier"])
        self.expect("frame variable 'static_cast<parent>(1)'",
                    error=True, substrs=["unknown type name 'parent'"])
        self.expect("frame variable 'static_cast<T_1<int> CxxParent>(1)'",
                    error=True, substrs=["two or more data types in declaration of 'type name'"])


        # TestReinterpretCast

        # Integers and enums can be converted to its own type.
        self.expect("frame variable 'reinterpret_cast<bool>(true)'", substrs=["true"])
        self.expect("frame variable 'reinterpret_cast<int>(5)'", substrs=["5"])
        self.expect("frame variable 'reinterpret_cast<td_int_t>(6)'", substrs=["6"])
        self.expect("frame variable 'reinterpret_cast<int>(td_int)'", substrs=["13"])
        self.expect("frame variable 'reinterpret_cast<long long>(100LL)'", substrs=["100"])
        self.expect("frame variable 'reinterpret_cast<UEnum>(u_enum)'", substrs=["kUTwo"])
        self.expect("frame variable 'reinterpret_cast<SEnum>(s_enum)'", substrs=["kSOne"])
        self.expect("frame variable 'reinterpret_cast<td_senum_t>(s_enum)'", substrs=["kSOne"])
        # Other scalar/enum to scalar/enum casts aren't allowed.
        self.expect("frame variable 'reinterpret_cast<int>(5U)'",
                    error=True, substrs=["reinterpret_cast from 'unsigned int' to 'int' is not allowed"])
        self.expect("frame variable 'reinterpret_cast<int>(3.14f)'",
                    error=True, substrs=["reinterpret_cast from 'float' to 'int' is not allowed"])
        self.expect("frame variable 'reinterpret_cast<double>(2.71)'",
                    error=True, substrs=["reinterpret_cast from 'double' to 'double' is not allowed"])
        self.expect("frame variable 'reinterpret_cast<int>(s_enum)'",
                    error=True, substrs=["reinterpret_cast from 'SEnum' to 'int' is not allowed"])
        self.expect("frame variable 'reinterpret_cast<UEnum>(0)'",
                    error=True, substrs=["reinterpret_cast from 'int' to 'UEnum' is not allowed"])
        self.expect("frame variable 'reinterpret_cast<UEnum>(s_enum)'",
                    error=True, substrs=["reinterpret_cast from 'SEnum' to 'UEnum' is not allowed"])

        # Pointers should be convertible to large enough integral types.
        #  self.expect("frame variable 'reinterpret_cast<long long>(ptr)'", IsOk[])
        #  self.expect("frame variable 'reinterpret_cast<long long>(arr)'", IsOk[])
        self.expect("frame variable 'reinterpret_cast<long long>(nullptr)'", substrs=["0"])
        if Is32Bit:
          pass
          #    self.expect("frame variable 'reinterpret_cast<int>(ptr)'", IsOk[])
          #    self.expect("frame variable 'reinterpret_cast<td_int_t>(ptr)'", IsOk[])
        else:
          self.expect("frame variable 'reinterpret_cast<int>(ptr)'",
                      error=True, substrs=["cast from pointer to smaller type 'int' loses information"])
          self.expect("frame variable 'reinterpret_cast<td_int_t>(ptr)'",
                      error=True, substrs=["cast from pointer to smaller type 'td_int_t' (canonically referred to as "
                                           "'int') loses information"])
        self.expect("frame variable 'reinterpret_cast<bool>(arr)'",
                    error=True, substrs=["cast from pointer to smaller type 'bool' loses information"])
        self.expect("frame variable 'reinterpret_cast<bool>(nullptr)'",
                    error=True, substrs=["cast from pointer to smaller type 'bool' loses information"])
        ##ifdef _WIN32
        #self.expect("frame variable 'reinterpret_cast<long>(ptr)'",
        #            error=True, substrs=["cast from pointer to smaller type 'long' loses information"])
        #else
        ##self.expect("frame variable 'reinterpret_cast<long>(ptr)'", IsOk[])
        #endif

        # Integers, enums and pointers can be converted to pointers.
        if Is32Bit:
          self.expect("frame variable 'reinterpret_cast<int*>(true)'",
                      substrs=["0x00000001"])
          self.expect("frame variable 'reinterpret_cast<float*>(6)'",
                      substrs=["0x00000006"])
          self.expect("frame variable 'reinterpret_cast<void*>(s_enum)'",
                      substrs=["0x00000001"])
          self.expect("frame variable 'reinterpret_cast<CxxBase*>(u_enum)'",
                      substrs=["0x00000002"])
          self.expect("frame variable '*reinterpret_cast<UEnum**>(ptr)'",
                      substrs=["0x00000001"])
        else:
          self.expect("frame variable 'reinterpret_cast<int*>(true)'",
                      substrs=["0x0000000000000001"])
          self.expect("frame variable 'reinterpret_cast<float*>(6)'",
                      substrs=["0x0000000000000006"])
          self.expect("frame variable 'reinterpret_cast<void*>(s_enum)'",
                      substrs=["0x0000000000000001"])
          self.expect("frame variable 'reinterpret_cast<CxxBase*>(u_enum)'",
                      substrs=["0x0000000000000002"])
          self.expect("frame variable '*reinterpret_cast<UEnum**>(ptr)'",
                      substrs=["0x0000000200000001"])

        #  self.expect("frame variable 'reinterpret_cast<td_int_ptr_t>(ptr)'", IsOk[])
        self.expect("frame variable '*reinterpret_cast<int*>(arr)'", substrs=["1"])
        self.expect("frame variable '*reinterpret_cast<long long*>(arr)'",
                    substrs=["8589934593"])  # 8589934593 == 0x0000000200000001

        # Casting to nullptr_t or nullptr_t to pointer types isn't allowed.
        ##ifndef __EMSCRIPTEN__
        #  EXPECT_THAT(
        #      Eval("reinterpret_cast<void*>(nullptr)'",
        #      error=True, substrs=["reinterpret_cast from 'nullptr_t' to 'void *' is not allowed"));
        #  self.expect("frame variable 'reinterpret_cast<std::nullptr_t>(ptr)'",
        #              error=True, substrs=["reinterpret_cast from 'int *' to 'std::nullptr_t' (aka #"
        #                      "'nullptr_t') is not allowed"])
        #  self.expect("frame variable 'reinterpret_cast<std::nullptr_t>(0)'",
        #              error=True, substrs=["reinterpret_cast from 'int' to 'std::nullptr_t' (aka "
        #                      "'nullptr_t') is not allowed"])
        #  self.expect("frame variable 'reinterpret_cast<std::nullptr_t>(nullptr)'",
        #              error=True, substrs=["reinterpret_cast from 'nullptr_t' to 'std::nullptr_t' "
        #                      "(aka 'nullptr_t') is not allowed"])
        ##else
        self.expect("frame variable 'reinterpret_cast<void*>(nullptr)'",
                    error=True, substrs=["reinterpret_cast from 'std::nullptr_t' to 'void *' is not "
                                         "allowed"])
        self.expect("frame variable 'reinterpret_cast<std::nullptr_t>(ptr)'",
                    error=True, substrs=["reinterpret_cast from 'int *' to 'std::nullptr_t' is not "
                                         "allowed"])
        self.expect("frame variable 'reinterpret_cast<std::nullptr_t>(0)'",
                    error=True, substrs=["reinterpret_cast from 'int' to 'std::nullptr_t' is "
                                         "not allowed"])
        self.expect("frame variable 'reinterpret_cast<std::nullptr_t>(nullptr)'",
                    error=True, substrs=["reinterpret_cast from 'std::nullptr_t' to "
                                         "'std::nullptr_t' "
                                         "is not allowed"])
        ##endif

        # L-values can be converted to reference type.
        self.expect("frame variable 'reinterpret_cast<CxxBase&>(arr[0]).a'", substrs=["1"])
        self.expect("frame variable 'reinterpret_cast<CxxBase&>(arr).b'", substrs=["2"])
        self.expect("frame variable 'reinterpret_cast<CxxParent&>(arr[0]).c'",
                    substrs=["17179869187"])  # 17179869187 == 0x0000000400000003
        self.expect("frame variable 'reinterpret_cast<CxxParent&>(arr).d'", substrs=["5"])
        #  self.expect("frame variable 'reinterpret_cast<int&>(parent)'", IsOk[])
        #  self.expect("frame variable 'reinterpret_cast<td_int_ref_t>(ptr)'", IsOk[])
        self.expect("frame variable 'reinterpret_cast<int&>(5)'",
                    error=True, substrs=["reinterpret_cast from rvalue to reference type 'int &'"])


        # Is result L-value or R-value?
        #  self.expect("frame variable '&reinterpret_cast<int&>(arr[0])'", IsOk[])
        self.expect("frame variable '&reinterpret_cast<int>(arr[0])'",
                    error=True, substrs=["cannot take the address of an rvalue of type 'int'"])
        self.expect("frame variable '&reinterpret_cast<UEnum>(u_enum)'",
                    error=True, substrs=["cannot take the address of an rvalue of type 'UEnum'"])
        self.expect("frame variable '&reinterpret_cast<int*>(arr)'",
                    error=True, substrs=["cannot take the address of an rvalue of type 'int *'"])

        # TestCxxDynamicCast

        error = process.Continue();
        threads = get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertEqual(
            len(threads), 1, "There should be a thread stopped at our breakpoint"
        )
        # The hit count for the breakpoint should be 2.
        self.assertEquals(breakpoint.GetHitCount(), 2)

        # LLDB doesn't support `dynamic_cast` in the expression evaluator.
        ##this->compare_with_lldb_ = false;

        self.expect("frame variable 'dynamic_cast<int>(0)'",
                    error=True, substrs=["invalid target type 'int' for dynamic_cast"])
        self.expect("frame variable 'dynamic_cast<int*>(0)'",
                    error=True, substrs=["'int' is not a class type"])
        self.expect("frame variable 'dynamic_cast<CxxBase*>(1.1)'",
              error=True, substrs=[
                  "cannot use dynamic_cast to convert from 'double' to 'CxxBase *'"])
        self.expect("frame variable 'dynamic_cast<CxxBase*>((int*)0)'",
                    error=True, substrs=["'int' is not a class type"])
        self.expect("frame variable 'dynamic_cast<CxxVirtualParent*>(base)'",
                    error=True, substrs=["'CxxBase' is not polymorphic"])
        self.expect("frame variable 'dynamic_cast<CxxVirtualParent*>(v_base)'",
                    error=True, substrs=["dynamic_cast is not supported in this context"])
