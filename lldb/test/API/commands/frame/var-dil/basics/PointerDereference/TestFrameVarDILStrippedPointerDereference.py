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

class TestFrameVarDILPointerDereference(TestBase):
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

        # Tests

        self.expect("settings set target.experimental.use-DIL true",
                    substrs=[""])
        self.expect("frame variable '*p_int0'", substrs=["0"])
        #self.expect("frame variable '*p_int0 + 1'", substrs=["1"])
        self.expect("frame variable '*cp_int5'", substrs=["5"])
        #self.expect("frame variable '*cp_int5 - 1'", substrs=["4"])

        self.expect("frame variable '&p_void[0]'", error=True,
                    substrs=["subscript of pointer to incomplete type 'void'"])
                    #substrs=["failed to use pointer as array for index 0 for \"(void *) p_void\""])
        #self.expect("frame variable '&*p_void'", patterns=["0x[0-9]+"])
        self.expect("frame variable '&pp_void0[2]'", patterns=["0x[0-9]+"])

        #self.expect("frame variable '**pp_int0'", substrs=["0"])
        #self.expect("frame variable '**pp_int0 + 1'", substrs=["1"])
        #self.expect("frame variable '&**pp_int0'", patterns=["0x[0-9]+"])
        #self.expect("frame variable '&**pp_int0 + 1'",
        #            patterns=["0x[0-9]+"])

        Is32Bit = False
        if process.GetAddressByteSize() == 4:
          Is32Bit = True;

        if Is32Bit:
          self.expect("frame variable '&*p_null'",
                      substrs=["0x00000000"])
          self.expect("frame variable '&p_null[4]'",
                      substrs=["0x00000010"])
          self.expect("frame variable '&*(int*)0'",
                      substrs=["0x00000000"])
          self.expect("frame variable '&((int*)0)[1]'",
                      substrs=["0x00000004"])
         # self.expect("frame variable '&(true ? *p_null : *p_null)'",
         #             substrs=["0x00000000"])

          #self.expect("frame variable '&(false ? *p_null : *p_null)'",
          #            substrs=["0x00000000"])
          #self.expect("frame variable '&*(true ? p_null : nullptr)'",
          #            substrs=["0x00000000"])
        else:
          pass
          #self.expect("frame variable '&*p_null'",
          #            substrs=["0x0000000000000000"])
          #self.expect("frame variable '&p_null[4]'",
          #            substrs=["0x0000000000000010"])
          #self.expect("frame variable '&*(int*)0'",
          #            substrs=["0x0000000000000000"])
          #self.expect("frame variable '&((int*)0)[1]'",
          #            substrs=["0x0000000000000004"])
          #self.expect("frame variable '&(true ? *p_null : *p_null)'",
          #            substrs=["0x0000000000000000"])

          #self.expect("frame variable '&(false ? *p_null : *p_null)'",
          #            substrs=["0x0000000000000000"])
          #self.expect("frame variable '&*(true ? p_null : nullptr)'",
          #            substrs=["0x0000000000000000"])
