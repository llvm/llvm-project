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

class TestFrameVarDILAddressOf(TestBase):
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
        self.expect("frame variable '&x'", patterns=["0x[0-9]+"])
        self.expect("frame variable 'r'", substrs=["42"])
        self.expect("frame variable '&r'", patterns=["0x[0-9]+"])
        self.expect("frame variable 'pr'", patterns=["0x[0-9]+"])
        self.expect("frame variable '&pr'", patterns=["0x[0-9]+"])
        self.expect("frame variable 'my_pr'", patterns=["0x[0-9]+"])
        self.expect("frame variable '&my_pr'", patterns=["0x[0-9]+"])

        #self.expect("frame variable '&x == &r'", substrs=["true"])
        #self.expect("frame variable '&x != &r'", substrs=["false"])

        #self.expect("frame variable '&p == &pr'", substrs=["true"])
        #self.expect("frame variable '&p != &pr'", substrs=["false"])
        #self.expect("frame variable '&p == &my_pr'", substrs=["true"])
        #self.expect("frame variable '&p != &my_pr'", substrs=["false"])

        self.expect("frame variable '&globalVar'", patterns=["0x[0-9]+"])
        self.expect("frame variable '&s_str'", patterns=["0x[0-9]+"])
        self.expect("frame variable '&param'", patterns=["0x[0-9]+"])

        #self.expect("frame variable '&(true ? x : x)'",
        #            patterns=["0x[0-9]+"])
        #self.expect("frame variable '&(true ? c : c)'",
        #            patterns=["0x[0-9]+"])

        self.expect("frame variable '&externGlobalVar'", error=True,
                    substrs=["use of undeclared identifier 'externGlobalVar'"])
                    #substrs=["no variable or instance variable named 'externGlobalVar' found in this frame"])

        self.expect("frame variable '&1'", error=True,
                    substrs=["cannot take the address of an rvalue of type "
                             "'int'"])
                    #substrs=["no variable or instance variable named '1' found in this frame"])

        self.expect("frame variable '&0.1'", error=True,
                    substrs=["cannot take the address of an rvalue of type "
                             "'double'"])
                    #substrs=["no variable or instance variable named '0' found in this frame"])

        #self.expect("frame variable '&(true ? 1 : 1)'", error=True,
        #            substrs=["cannot take the address of an rvalue of type "
        #                     "'int'"])

        #self.expect("frame variable '&(true ? c : (char)1)'", error=True,
        #            substrs=["cannot take the address of an rvalue of type "
        #                     "'char'"])
        #self.expect("frame variable '&(true ? c : 1)'", error=True,
        #            substrs=["cannot take the address of an rvalue of type "
        #                     "'int'"])

        #self.expect("frame variable '&this'", error=True,
        #            substrs=["cannot take the address of an rvalue of type "
        #                     "'TestMethods *'"])
        self.expect("frame variable '&this'", error=True,
                    substrs=["cannot take the address of an rvalue of type "
                             "'TestMethods *'"])
                    #patterns=["0x[0-9]+"])

        self.expect("frame variable '&(&s_str)'", error=True,
                    substrs=["cannot take the address of an rvalue of type "
                             "'const char **'"])
                    #substrs=["no variable or instance variable named '(' found in this frame"])
