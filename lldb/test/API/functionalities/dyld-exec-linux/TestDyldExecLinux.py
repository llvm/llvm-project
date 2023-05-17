"""
Test that LLDB can launch a linux executable and then execs into the dynamic
loader into this program again.
"""

import lldb
import os

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestLinux64ExecViaDynamicLoader(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfXmlSupportMissing
    @skipIf(oslist=no_match(['linux']))
    def test_with_svr4(self):
        self.runCmd("settings set plugin.process.gdb-remote.use-libraries-svr4 true")
        self._test()

    @skipIf(oslist=no_match(['linux']))
    def test_without_svr4(self):
        self.runCmd("settings set plugin.process.gdb-remote.use-libraries-svr4 false")
        self._test()

    def _test(self):
        self.build()

        # Extracts path of the interpreter.
        exe = self.getBuildArtifact("a.out")

        spec = lldb.SBModuleSpec()
        spec.SetFileSpec(lldb.SBFileSpec(exe))
        interp_section = lldb.SBModule(spec).FindSection(".interp")
        if not interp_section:
          return
        section_data = interp_section.GetSectionData()
        error = lldb.SBError()
        dyld_path = section_data.GetString(error,0)
        if error.Fail():
          return

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set a breakpoint in the main function that will get hit after the
        # program exec's via the dynamic loader. The breakpoint will only get
        # hit if we can successfully read the shared library lists in the
        # DynamicLoaderPOSIXDYLD.cpp when we exec into the dynamic loader.
        breakpoint_main = target.BreakpointCreateBySourceRegex("// Break here", lldb.SBFileSpec("main.cpp"))
        # Setup our launch info to supply the dynamic loader path to the
        # program so it gets two args:
        # - path to a.out
        # - path to dynamic loader
        launch_info = lldb.SBLaunchInfo([dyld_path])
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertSuccess(error)

        threads = lldbutil.get_stopped_threads(process, lldb.eStopReasonExec)
        self.assertEqual(len(threads), 1, "We got a thread stopped for exec.")

        process.Continue();

        # Stopped on main here.
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = process.GetSelectedThread()
        self.assertIn("main", thread.GetFrameAtIndex(0).GetDisplayFunctionName())
