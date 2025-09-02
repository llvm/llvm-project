"""
Test that stop hooks fire on core load (first stop)
"""


import lldb
import os
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestStopOnCoreLoad(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # This was originally marked as expected failure on Windows, but it has
    # started timing out instead, so the expectedFailure attribute no longer
    # correctly tracks it: llvm.org/pr37371
    @skipIfWindows
    def test_hook_runs_no_threads(self):
        # Create core form YAML.
        core_path = self.getBuildArtifact("test.core")
        self.yaml2obj("test.core.yaml", core_path)

        # Since mach core files don't have stop reasons, we should choose
        # the first thread:
        self.do_test(core_path, 1)

    def test_hook_one_thread(self):
        core_path = os.path.join(self.getSourceDir(), "linux-x86_64.core")
        self.do_test(core_path, 3)

    def do_test(self, core_path, stop_thread):
        # Set debugger into synchronous mode
        self.dbg.SetAsync(False)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget("")

        # load the stop hook module and add the stop hook:
        stop_hook_path = os.path.join(self.getSourceDir(), "stop_hook.py")
        self.runCmd(f"command script import {stop_hook_path}")
        self.runCmd("target stop-hook add -P stop_hook.stop_handler")

        # Load core.
        process = target.LoadCore(core_path)
        self.assertTrue(process, PROCESS_IS_VALID)
        # Now run our report command and make sure we get the right answer.

        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand("report_command", result)
        print(f"Command Output: '{result.GetOutput}'")
        self.assertIn(
            f"Stop Threads: {stop_thread}", result.GetOutput(), "Ran the stop hook"
        )
