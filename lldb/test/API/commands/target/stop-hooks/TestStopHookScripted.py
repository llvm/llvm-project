"""
Test stop hook functionality
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

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
        
    def test_bad_handler(self):
        """Test that we give a good error message when the handler is bad"""
        self.script_setup()
        result = lldb.SBCommandReturnObject()

        # First try the wrong number of args handler:
        command = "target stop-hook add -P stop_hook.bad_handle_stop"
        self.interp.HandleCommand(command, result)
        self.assertFalse(result.Succeeded(), "Set the target stop hook")
        self.assertIn("Wrong number of args", result.GetError(), "Got the wrong number of args error")

        # Next the no handler at all handler:
        command = "target stop-hook add -P stop_hook.no_handle_stop"
            
        self.interp.HandleCommand(command, result)
        self.assertFalse(result.Succeeded(), "Set the target stop hook")
        self.assertIn('Class "stop_hook.no_handle_stop" is missing the required handle_stop callback', result.GetError(), "Got the right error")
        
    def test_stop_hooks_scripted(self):
        """Test that a scripted stop hook works with no specifiers"""
        self.stop_hooks_scripted(5)

    def test_stop_hooks_scripted_right_func(self):
        """Test that a scripted stop hook fires when there is a function match"""
        self.stop_hooks_scripted(5, "-n step_out_of_me")
    
    def test_stop_hooks_scripted_wrong_func(self):
        """Test that a scripted stop hook doesn't fire when the function does not match"""
        self.stop_hooks_scripted(0, "-n main")
    
    def test_stop_hooks_scripted_right_lines(self):
        """Test that a scripted stop hook fires when there is a function match"""
        self.stop_hooks_scripted(5, "-f main.c -l 1 -e %d"%(self.main_start_line))
    
    def test_stop_hooks_scripted_wrong_lines(self):
        """Test that a scripted stop hook doesn't fire when the function does not match"""
        self.stop_hooks_scripted(0, "-f main.c -l %d -e 100"%(self.main_start_line))

    def test_stop_hooks_scripted_auto_continue(self):
        """Test that the --auto-continue flag works"""
        self.do_test_auto_continue(False)

    def test_stop_hooks_scripted_return_false(self):
        """Test that the returning False from a stop hook works"""
        self.do_test_auto_continue(True)

    def do_test_auto_continue(self, return_true):
        """Test that auto-continue works."""
        # We set auto-continue to 1 but the stop hook only applies to step_out_of_me,
        # so we should end up stopped in main, having run the expression only once.
        self.script_setup()
        
        result = lldb.SBCommandReturnObject()

        if return_true:
          command = "target stop-hook add -P stop_hook.stop_handler -k increment -v 5 -k return_false -v 1 -n step_out_of_me"
        else:
          command = "target stop-hook add -G 1 -P stop_hook.stop_handler -k increment -v 5 -n step_out_of_me"
            
        self.interp.HandleCommand(command, result)
        self.assertTrue(result.Succeeded(), "Set the target stop hook")

        # First run to main.  If we go straight to the first stop hook hit,
        # run_to_source_breakpoint will fail because we aren't at original breakpoint

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Stop here first", self.main_source_file)

        # Now set the breakpoint on step_out_of_me, and make sure we run the
        # expression, then continue back to main.
        bkpt = target.BreakpointCreateBySourceRegex("Set a breakpoint here and step out", self.main_source_file)
        self.assertNotEqual(bkpt.GetNumLocations(), 0, "Got breakpoints in step_out_of_me")
        process.Continue()

        var = target.FindFirstGlobalVariable("g_var")
        self.assertTrue(var.IsValid())
        self.assertEqual(var.GetValueAsUnsigned(), 6, "Updated g_var")
        
        func_name = process.GetSelectedThread().frames[0].GetFunctionName()
        self.assertEqual("main", func_name, "Didn't stop at the expected function.")

    def script_setup(self):
        self.interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()

        # Bring in our script file:
        script_name = os.path.join(self.getSourceDir(), "stop_hook.py")
        command = "command script import " + script_name
        self.interp.HandleCommand(command, result)
        self.assertTrue(result.Succeeded(), "com scr imp failed: %s"%(result.GetError()))

        # set a breakpoint at the end of main to catch our auto-continue tests.
        # Do it in the dummy target so it will get copied to our target even when
        # we don't have a chance to stop.
        dummy_target = self.dbg.GetDummyTarget()
        dummy_target.BreakpointCreateBySourceRegex("return result", self.main_source_file)

        
    def stop_hooks_scripted(self, g_var_value, specifier = None):
        self.script_setup()
        
        result = lldb.SBCommandReturnObject()

        command = "target stop-hook add -P stop_hook.stop_handler -k increment -v 5 "
        if specifier:
            command += specifier
        
        self.interp.HandleCommand(command, result)
        self.assertTrue(result.Succeeded(), "Set the target stop hook")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file)
        # At this point we've hit our stop hook so we should have run our expression,
        # which increments g_var by the amount specified by the increment key's value.
        while process.GetState() == lldb.eStateRunning:
            continue

        var = target.FindFirstGlobalVariable("g_var")
        self.assertTrue(var.IsValid())
        self.assertEqual(var.GetValueAsUnsigned(), g_var_value, "Updated g_var")
