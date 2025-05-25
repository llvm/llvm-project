"""
Test that a Python breakpoint callback defined in another Python
breakpoint callback works properly. 
"""


import lldb
import os
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestNestedBreakpointCommands(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_nested_commands(self):
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.callback_module = "make_bkpt_cmds"
        self.do_test()

    def do_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source_file
        )

        outer_bkpt = target.BreakpointCreateBySourceRegex(
            "Set outer breakpoint here", self.main_source_file
        )
        cmd_file_path = os.path.join(self.getSourceDir(), f"{self.callback_module}.py")
        self.runCmd(f"command script import {cmd_file_path}")
        outer_bkpt.SetScriptCallbackFunction(f"{self.callback_module}.outer_callback")

        process.Continue()

        self.assertEqual(
            thread.stop_reason, lldb.eStopReasonBreakpoint, "Right stop reason"
        )

        bkpt_no = thread.stop_reason_data[0]

        # We made the callbacks record the new breakpoint ID and the number of
        # times a callback ran in some globals in the target.  Find them now:
        exec_module = target.FindModule(target.executable)
        self.assertTrue(exec_module.IsValid(), "Found executable module")
        var = exec_module.FindFirstGlobalVariable(target, "g_global")
        self.assertSuccess(var.GetError(), "Found globals")
        num_hits = var.GetChildAtIndex(1).GetValueAsUnsigned()
        inner_id = var.GetChildAtIndex(2).GetValueAsUnsigned()

        # Make sure they have the right values:
        self.assertEqual(bkpt_no, inner_id, "Hit the right breakpoint")
        self.assertEqual(num_hits, 2, "Right counter end value")
        self.assertEqual(thread.frames[0].name, "main", "Got to main")

        self.assertEqual(outer_bkpt.GetHitCount(), 1, "Hit outer breakpoint once")

        inner_bkpt = target.FindBreakpointByID(inner_id)
        self.assertEqual(inner_bkpt.GetHitCount(), 1, "Hit inner breakpoint once")
