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


class TestFrameVarDILGlobalVariableLookup(TestBase):
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
        self.assertEqual(breakpoint.GetHitCount(), 1)

        frame = threads[0].GetFrameAtIndex(0)
        command_result = lldb.SBCommandReturnObject()
        interp = self.dbg.GetCommandInterpreter()

        self.expect("settings set target.experimental.use-DIL true", substrs=[""])
        self.expect("frame variable 'globalVar'", substrs=["-559038737"])  # 0xDEADBEEF
        self.expect("frame variable 'globalPtr'", patterns=["0x[0-9]+"])
        self.expect("frame variable 'globalRef'", substrs=["-559038737"])
        self.expect("frame variable '::globalPtr'", patterns=["0x[0-9]+"])
        self.expect("frame variable '::globalRef'", substrs=["-559038737"])

        self.expect(
            "frame variable 'externGlobalVar'",
            error=True,
            substrs=["use of undeclared identifier"],
        )  # 0x00C0FFEE
        self.expect(
            "frame variable '::externGlobalVar'",
            error=True,
            substrs=["use of undeclared identifier"],
        )  # ["12648430"])
        self.expect("frame variable 'ns::globalVar'", substrs=["13"])
        self.expect("frame variable 'ns::globalPtr'", patterns=["0x[0-9]+"])
        self.expect("frame variable 'ns::globalRef'", substrs=["13"])
        self.expect("frame variable '::ns::globalVar'", substrs=["13"])
        self.expect("frame variable '::ns::globalPtr'", patterns=["0x[0-9]+"])
