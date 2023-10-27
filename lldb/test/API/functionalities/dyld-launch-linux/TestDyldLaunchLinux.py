"""
Test that LLDB can launch a linux executable through the dynamic loader and still hit a breakpoint.
"""

import lldb
import os

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestLinux64LaunchingViaDynamicLoader(TestBase):
    @skipIf(oslist=no_match(["linux"]))
    @no_debug_info_test
    @skipIf(oslist=["linux"], archs=["arm"])
    def test(self):
        self.build()

        # Extracts path of the interpreter.
        spec = lldb.SBModuleSpec()
        spec.SetFileSpec(lldb.SBFileSpec(self.getBuildArtifact("a.out")))
        interp_section = lldb.SBModule(spec).FindSection(".interp")
        if not interp_section:
            return
        section_data = interp_section.GetSectionData()
        error = lldb.SBError()
        exe = section_data.GetString(error, 0)
        if error.Fail():
            return

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set breakpoints both on shared library function as well as on
        # main. Both of them will be pending breakpoints.
        breakpoint_main = target.BreakpointCreateBySourceRegex(
            "// Break here", lldb.SBFileSpec("main.cpp")
        )
        breakpoint_shared_library = target.BreakpointCreateBySourceRegex(
            "get_signal_crash", lldb.SBFileSpec("signal_file.cpp")
        )
        launch_info = lldb.SBLaunchInfo(
            [
                "--library-path",
                self.get_process_working_directory(),
                self.getBuildArtifact("a.out"),
            ]
        )
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertSuccess(error)

        # Stopped on main here.
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = process.GetSelectedThread()
        self.assertIn("main", thread.GetFrameAtIndex(0).GetDisplayFunctionName())
        process.Continue()

        # Stopped on get_signal_crash function here.
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertIn(
            "get_signal_crash", thread.GetFrameAtIndex(0).GetDisplayFunctionName()
        )
        process.Continue()

        # Stopped because of generated signal.
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertIn("raise", lldbutil.get_function_names(thread))
