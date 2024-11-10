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

class TestFrameVarDILBitwiseOperators(TestBase):
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

        #self.expect("settings set target.experimental.use-DIL true",
        #            substrs=[""])
        #
        # TestBitwiseOperators
        #
        #self.expect("frame variable '~(-1)'", substrs=["0"])
        #self.expect("frame variable '~~0'", substrs=["0"])
        #self.expect("frame variable '~0'", substrs=["-1"])
        #self.expect("frame variable '~1'", substrs=["-2"])
        #self.expect("frame variable '~0LL'", substrs=["-1"])
        #self.expect("frame variable '~1LL'", substrs=["-2"])
        #self.expect("frame variable '~true'", substrs=["-2"])
        #self.expect("frame variable '~false'", substrs=["-1"])
        #self.expect("frame variable '~var_true'", substrs=["-2"])
        #self.expect("frame variable '~var_false'", substrs=["-1"])
        #self.expect("frame variable '~ull_max'", substrs=["0"])
        #self.expect("frame variable '~ull_zero'", substrs=["18446744073709551615"])

        #self.expect("frame variable '~s'", error=True,
        #            substrs=["invalid argument type 'S' to unary expression"])
        #self.expect("frame variable '~p'", error=True,
        #            substrs=["invalid argument type 'const char *' to unary expression"])

        #self.expect("frame variable '(1 << 5)'", substrs=["32"])
        #self.expect("frame variable '(32 >> 2)'", substrs=["8"])
        #self.expect("frame variable '(-1 >> 10)'", substrs=["-1"])
        #self.expect("frame variable '(-100 >> 5)'", substrs=["-4"])
        #self.expect("frame variable '(-3 << 6)'", substrs=["-192"])
        #self.expect("frame variable '(2000000000U << 1)'", substrs=["4000000000"])
        #self.expect("frame variable '(-1 >> 1U)'", substrs=["-1"])
        #self.expect("frame variable '(char)1 << 16'", substrs=["65536"])
        #self.expect("frame variable '(signed char)-123 >> 8'", substrs=["-1"])

        #self.expect("frame variable '0b1011 & 0xFF'", substrs=["11"])
        #self.expect("frame variable '0b1011 & mask_ff'", substrs=["11"])
        #self.expect("frame variable '0b1011 & 0b0111'", substrs=["3"])
        #self.expect("frame variable '0b1011 | 0b0111'", substrs=["15"])
        #self.expect("frame variable -- '-0b1011 | 0xFF'", substrs=["-1"])
        #self.expect("frame variable -- '-0b1011 | 0xFFu'", substrs=["4294967295"])
        #self.expect("frame variable '0b1011 ^ 0b0111'", substrs=["12"])
        #self.expect("frame variable '~0b1011'", substrs=["-12"])
