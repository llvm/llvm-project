"""
Test that LLDB can launch a linux executable through the dynamic loader and still hit a breakpoint.
"""

import lldb
import os

from lldbsuite.test.lldbtest import *

class TestLinux64LaunchingViaDynamicLoader(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        self.build()
        exe = "/lib64/ld-linux-x86-64.so.2"
        if(os.path.exists(exe)):
            target = self.dbg.CreateTarget(exe)
            self.assertTrue(target, VALID_TARGET)

            # Set breakpoints both on shared library function as well as on
            # main. Both of them will be pending breakpoints.
            breakpoint_main = target.BreakpointCreateBySourceRegex("// Break here", lldb.SBFileSpec("main.cpp"))
            breakpoint_shared_library = target.BreakpointCreateBySourceRegex("get_signal_crash", lldb.SBFileSpec("signal_file.cpp"))
            launch_info = lldb.SBLaunchInfo([ "--library-path",self.get_process_working_directory(),self.getBuildArtifact("a.out")])
            launch_info.SetWorkingDirectory(self.get_process_working_directory())
            error = lldb.SBError()
            process = target.Launch(launch_info,error)
            self.assertTrue(error.Success())

            # Stopped on main here.
            self.assertEqual(process.GetState(), lldb.eStateStopped)
            thread = process.GetSelectedThread()
            self.assertIn("main",thread.GetFrameAtIndex(0).GetDisplayFunctionName())
            process.Continue()

            # Stopped on get_signal_crash function here.
            self.assertEqual(process.GetState(), lldb.eStateStopped)
            self.assertIn("get_signal_crash",thread.GetFrameAtIndex(0).GetDisplayFunctionName())
            process.Continue()

            # Stopped because of generated signal.
            self.assertEqual(process.GetState(), lldb.eStateStopped)
            self.assertIn("raise",thread.GetFrameAtIndex(0).GetDisplayFunctionName())
            self.assertIn("get_signal_crash",thread.GetFrameAtIndex(1).GetDisplayFunctionName())


