"""
Test that pending breakpoints resolve for JITted code with mcjit and rtdyld.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

import shutil


class TestJitBreakpoint(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.ll = self.getBuildArtifact("jitbp.ll")

    @skipUnlessArch("x86_64")
    @expectedFailureAll(oslist=["windows"])
    def test_jit_breakpoints(self):
        self.build()
        self.do_test("--jit-kind=mcjit")
        self.do_test("--jit-linker=rtdyld")

    def do_test(self, jit_flag: str):
        self.dbg.SetAsync(False)

        self.dbg.HandleCommand("settings set plugin.jit-loader.gdb.enable on")

        lldb_dir = os.path.dirname(lldbtest_config.lldbExec)
        lli_path = shutil.which("lli", path=lldb_dir)
        self.assertTrue(os.path.exists(lli_path), "lli not found")
        target = self.dbg.CreateTarget(lli_path)
        self.assertTrue(target.IsValid())

        bp = target.BreakpointCreateByName("jitbp")
        self.assertTrue(bp.IsValid())
        self.assertEqual(bp.GetNumLocations(), 0, "Expected a pending breakpoint")

        launch_info = target.GetLaunchInfo()
        launch_info.SetArguments([jit_flag, self.ll], True)

        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(process.IsValid())
        self.assertTrue(error.Success(), error.GetCString())

        self.assertEqual(process.GetState(), lldb.eStateStopped)

        thread = process.GetSelectedThread()
        frame = thread.GetSelectedFrame()
        self.assertIn("jitbp", frame.GetFunctionName())

        self.assertGreaterEqual(
            bp.GetNumLocations(), 1, "Breakpoint must be resolved after JIT loads code"
        )
