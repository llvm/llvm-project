"""
Make sure 'frame var' using DIL parser/evaultor works for local variables.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil

import os
import shutil
import time

class TestFrameVarDILSubscript(TestBase):
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

        self.expect("settings set target.experimental.use-DIL true",
                    substrs=[""])

        # const char*
        self.expect("frame variable 'char_ptr[0]'", substrs=["'l'"])
        #self.expect("frame variable '1[char_ptr]'", substrs=["'o'"])

        # const char[]
        self.expect("frame variable 'char_arr[0]'", substrs=["'i'"])
        #self.expect("frame variable '1[char_arr]'", substrs=["'p'"])

        # Boolean types are integral too!
        #self.expect("frame variable 'int_arr[false]'", substrs=["1"])
        #self.expect("frame variable 'true[int_arr]'", substrs=["2"])

        # As well as unscoped enums.
        #self.expect("frame variable 'int_arr[enum_one]'", substrs=["2"])
        #self.expect("frame variable 'enum_one[int_arr]'", substrs=["2"])

        # But floats are not.
        self.expect("frame variable 'int_arr[1.0]'", error=True,
                    substrs=["array subscript is not an integer"])
                    #substrs=["invalid range expression \"'1.0'\""])

        # Base should be a "pointer to T" and index should be of an integral
        # type.
        self.expect("frame variable 'char_arr[char_ptr]'", error=True,
                    substrs=["array subscript is not an integer"])
                    #substrs=["invalid index expression \"char_ptr\""])
        self.expect("frame variable '1[2]'", error=True,
                    substrs=["subscripted value is not an array or pointer"])
                    #substrs=["no variable named '1' found in this frame"])

        # Test when base and index are references.
        self.expect("frame variable 'c_arr[0].field_'", substrs=["0"])
        #self.expect("frame variable 'c_arr[idx_1_ref].field_'",
        #            substrs=["1"])
        #self.expect("frame variable 'c_arr[enum_ref].field_'",
        #            substrs=["1"])
        #self.expect("frame variable 'c_arr_ref[0].field_'",
        #            substrs=["0"])
        #self.expect("frame variable 'c_arr_ref[idx_1_ref].field_'",
        #            substrs=["1"])
        #self.expect("frame variable 'c_arr_ref[enum_ref].field_'",
        #            substrs=["1"])

        self.expect("frame variable 'td_int_arr[0]'", substrs=["1"])
        #self.expect("frame variable 'td_int_arr[td_int_idx_1]'",
        #            substrs=["2"])
        #self.expect("frame variable 'td_int_arr[td_td_int_idx_2]'",
        #            substrs=["3"])
        self.expect("frame variable 'td_int_ptr[0]'", substrs=["1"])
        #self.expect("frame variable 'td_int_ptr[td_int_idx_1]'",
        #            substrs=["2"])
        #self.expect("frame variable 'td_int_ptr[td_td_int_idx_2]'",
        #            substrs=["3"])
        # Both typedefs and refs!
        #self.expect("frame variable 'td_int_arr_ref[td_int_idx_1_ref]'",
        #            substrs=["2"])

        # Test for index out of bounds.
        self.expect("frame variable 'int_arr[42]'", patterns=["[0-9]+"])
        self.expect("frame variable 'int_arr[100]'", patterns=["[0-9]+"])

        # Test for negative index.
        self.expect("frame variable 'int_arr[-1]'", patterns=["[0-9]+"])
        self.expect("frame variable 'int_arr[-42]'", patterns=["[0-9]+"])

        # Test for "max unsigned char".
        #self.expect("frame variable 'uint8_arr[uchar_idx]'",
        #            substrs=["'\\xab'"])

        # Test address-of of the subscripted value.
        #self.expect("frame variable '(&c_arr[1])->field_'", substrs=["1"])
