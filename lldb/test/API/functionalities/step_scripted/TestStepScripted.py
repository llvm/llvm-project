"""
Tests stepping with scripted thread plans.
"""
import threading
import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *

class StepScriptedTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.runCmd("command script import Steps.py")

    def test_standard_step_out(self):
        """Tests stepping with the scripted thread plan laying over a standard 
        thread plan for stepping out."""
        self.build()
        self.step_out_with_scripted_plan("Steps.StepOut")

    def test_scripted_step_out(self):
        """Tests stepping with the scripted thread plan laying over an another 
        scripted thread plan for stepping out."""
        self.build()
        self.step_out_with_scripted_plan("Steps.StepScripted")

    def step_out_with_scripted_plan(self, name):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                                                            "Set a breakpoint here",
                                                                            self.main_source_file)

        frame = thread.GetFrameAtIndex(0)
        self.assertEqual("foo", frame.GetFunctionName())

        err = thread.StepUsingScriptedThreadPlan(name)
        self.assertSuccess(err)

        frame = thread.GetFrameAtIndex(0)
        self.assertEqual("main", frame.GetFunctionName())


    def test_misspelled_plan_name(self):
        """Test that we get a useful error if we misspell the plan class name"""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                                                            "Set a breakpoint here",
                                                                            self.main_source_file)
        stop_id = process.GetStopID()
        # Pass a non-existent class for the plan class:
        err = thread.StepUsingScriptedThreadPlan("NoSuchModule.NoSuchPlan")
        
        # Make sure we got a good error:
        self.assertTrue(err.Fail(), "We got a failure state")
        msg = err.GetCString()
        self.assertIn("NoSuchModule.NoSuchPlan", msg, "Mentioned missing class")
        
        # Make sure we didn't let the process run:
        self.assertEqual(stop_id, process.GetStopID(), "Process didn't run")
        
    def test_checking_variable(self):
        """Test that we can call SBValue API's from a scripted thread plan - using SBAPI's to step"""
        self.do_test_checking_variable(False)
        
    def test_checking_variable_cli(self):
        """Test that we can call SBValue API's from a scripted thread plan - using cli to step"""
        self.do_test_checking_variable(True)
        
    def do_test_checking_variable(self, use_cli):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                                                            "Set a breakpoint here",
                                                                            self.main_source_file)

        frame = thread.GetFrameAtIndex(0)
        self.assertEqual("foo", frame.GetFunctionName())
        foo_val = frame.FindVariable("foo")
        self.assertSuccess(foo_val.GetError(), "Got the foo variable")
        self.assertEqual(foo_val.GetValueAsUnsigned(), 10, "foo starts at 10")

        if use_cli:
            result = lldb.SBCommandReturnObject()
            self.dbg.GetCommandInterpreter().HandleCommand(
                "thread step-scripted -C Steps.StepUntil -k variable_name -v foo",
                result)
            self.assertTrue(result.Succeeded())
        else:
            args_data = lldb.SBStructuredData()
            data = lldb.SBStream()
            data.Print('{"variable_name" : "foo"}')
            error = args_data.SetFromJSON(data)
            self.assertSuccess(error, "Made the args_data correctly")

            err = thread.StepUsingScriptedThreadPlan("Steps.StepUntil", args_data, True)
            self.assertSuccess(err)

        # We should not have exited:
        self.assertState(process.GetState(), lldb.eStateStopped, "We are stopped")

        # We should still be in foo:
        self.assertEqual("foo", frame.GetFunctionName())

        # And foo should have changed:
        self.assertTrue(foo_val.GetValueDidChange(), "Foo changed")

    def test_stop_others_from_command(self):
        """Test that the stop-others flag is set correctly by the command line.
           Also test that the run-all-threads property overrides this."""
        self.do_test_stop_others()

    def run_step(self, stop_others_value, run_mode, token):
        import Steps
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()

        cmd = "thread step-scripted -C Steps.StepReportsStopOthers -k token -v %s"%(token)
        if run_mode != None:
            cmd = cmd + " --run-mode %s"%(run_mode)
        print(cmd)
        interp.HandleCommand(cmd, result)
        self.assertTrue(result.Succeeded(), "Step scripted failed: %s."%(result.GetError()))
        print(Steps.StepReportsStopOthers.stop_mode_dict)
        value = Steps.StepReportsStopOthers.stop_mode_dict[token]
        self.assertEqual(value, stop_others_value, "Stop others has the correct value.")
        
    def do_test_stop_others(self):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                                                            "Set a breakpoint here",
                                                                            self.main_source_file)
        # First run with stop others false and see that we got that.
        thread_id = ""
        if sys.version_info.major == 2:
            thread_id = str(threading._get_ident())
        else:
            thread_id = str(threading.get_ident())

        # all-threads should set stop others to False.
        self.run_step(False, "all-threads", thread_id)

        # this-thread should set stop others to True
        self.run_step(True, "this-thread", thread_id)

        # The default value should be stop others:
        self.run_step(True, None, thread_id)

        # The target.process.run-all-threads should override this:
        interp = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        
        interp.HandleCommand("settings set target.process.run-all-threads true", result)
        self.assertTrue(result.Succeeded, "setting run-all-threads works.")

        self.run_step(False, None, thread_id)

        
        

        
