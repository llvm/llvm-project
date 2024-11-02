"""
Test stop hook functionality
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestStopHooks(TestBase):

    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        full_path = os.path.join(self.getSourceDir(), "main.c")
        self.main_start_line = line_number(full_path, "main()")
        
    def test_stop_hooks_step_out(self):
        """Test that stop hooks fire on step-out."""
        self.step_out_test()

    def test_stop_hooks_after_expr(self):
        """Test that a stop hook fires when hitting a breakpoint
           that runs an expression"""
        self.after_expr_test()

    def step_out_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file)

        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand("target stop-hook add -o 'expr g_var++'", result)
        self.assertTrue(result.Succeeded, "Set the target stop hook")
        thread.StepOut()
        var = target.FindFirstGlobalVariable("g_var")
        self.assertTrue(var.IsValid())
        self.assertEqual(var.GetValueAsUnsigned(), 1, "Updated g_var")

    def after_expr_test(self):
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand("target stop-hook add -o 'expr g_var++'", result)
        self.assertTrue(result.Succeeded, "Set the target stop hook")

        (target, process, thread, first_bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file)

        var = target.FindFirstGlobalVariable("g_var")
        self.assertTrue(var.IsValid())
        self.assertEqual(var.GetValueAsUnsigned(), 1, "Updated g_var")

        bkpt = target.BreakpointCreateBySourceRegex("Continue to here", self.main_source_file)
        self.assertNotEqual(bkpt.GetNumLocations(), 0, "Set the second breakpoint")
        commands = lldb.SBStringList()
        commands.AppendString("expr increment_gvar()")
        bkpt.SetCommandLineCommands(commands);
        
        threads = lldbutil.continue_to_breakpoint(process, bkpt)
        self.assertEqual(len(threads), 1, "Hit my breakpoint")
        
        self.assertTrue(var.IsValid())
        self.assertEqual(var.GetValueAsUnsigned(), 3, "Updated g_var")

        # Make sure running an expression does NOT run the stop hook.
        # Our expression will increment it by one, but the stop shouldn't
        # have gotten it to 5.
        threads[0].frames[0].EvaluateExpression("increment_gvar()")
        self.assertTrue(var.IsValid())
        self.assertEqual(var.GetValueAsUnsigned(), 4, "Updated g_var")
        

        # Make sure a rerun doesn't upset the state we've set up:
        process.Kill()
        lldbutil.run_to_breakpoint_do_run(self, target, first_bkpt)
        var = target.FindFirstGlobalVariable("g_var")
        self.assertTrue(var.IsValid())
        self.assertEqual(var.GetValueAsUnsigned(), 1, "Updated g_var")
        
        
