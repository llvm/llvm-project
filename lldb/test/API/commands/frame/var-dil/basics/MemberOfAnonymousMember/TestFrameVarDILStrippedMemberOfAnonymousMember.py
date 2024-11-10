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

class TestFrameVarDILMemberOfAnonymousMember(TestBase):
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
        self.expect("frame variable 'a.x'", substrs=["1"])
        self.expect("frame variable 'a.y'", substrs=["2"])

        #self.expect("frame variable 'b.x'", error=True,
        #            substrs=["no member named 'x' in 'B'"])
        self.expect("frame variable 'b.y'", error=True,
                    substrs=["no member named 'y' in 'B'"])
                    #substrs=["0"])
        self.expect("frame variable 'b.z'", substrs=["3"])
        self.expect("frame variable 'b.w'", substrs=["4"])
        self.expect("frame variable 'b.a.x'", substrs=["1"])
        self.expect("frame variable 'b.a.y'", substrs=["2"])

        self.expect("frame variable 'c.x'", substrs=["5"])
        self.expect("frame variable 'c.y'", substrs=["6"])

        self.expect("frame variable 'd.x'", substrs=["7"])
        self.expect("frame variable 'd.y'", substrs=["8"])
        self.expect("frame variable 'd.z'", substrs=["9"])
        self.expect("frame variable 'd.w'", substrs=["10"])

        self.expect("frame variable 'e.x'", error=True,
                    substrs=["no member named 'x' in 'E'"])
                    #substrs=["\"x\" is not a member of \"(E) e\""])
        self.expect("frame variable 'f.x'", error=True,
                    substrs=["no member named 'x' in 'F'"])
                    #substrs=["\"x\" is not a member of \"(F) f\""])
        self.expect("frame variable 'f.named_field.x'", substrs=["12"])

        #self.expect("frame variable 'unnamed_derived.x'", substrs=["1"])
        self.expect("frame variable 'unnamed_derived.y'", substrs=["2"])
        self.expect("frame variable 'unnamed_derived.z'",
                    substrs=["13"])
                    #substrs=["(A::(anonymous struct)) unnamed_derived.z = (x = 1)"])

        self.expect("frame variable 'derb.x'", error=True,
                    substrs=["no member named 'x' in 'DerivedB'"])
                    #substrs=["\"x\" is not a member of \"(DerivedB) derb\""])
        self.expect("frame variable 'derb.y'", error=True,
                    substrs=["no member named 'y' in 'DerivedB'"])
                    #substrs=["\"y\" is not a member of \"(DerivedB) derb\""])
        #self.expect("frame variable 'derb.z'", substrs=["3"])
        self.expect("frame variable 'derb.w'",
                    substrs=["14"])
                    #substrs=["z = 3"])
        self.expect("frame variable 'derb.k'",
                    substrs=["15"])
                    #substrs=["4"])
        self.expect("frame variable 'derb.a.x'", substrs=["1"])
        self.expect("frame variable 'derb.a.y'", substrs=["2"])
