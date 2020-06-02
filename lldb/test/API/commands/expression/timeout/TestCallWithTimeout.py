"""
Test calling a function that waits a while, and make sure the timeout option to expr works.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCommandWithTimeoutsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.main_source = "wait-a-while.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    @expectedFlakeyFreeBSD("llvm.org/pr19605")
    @expectedFailureAll(
        oslist=[
            "windows"],
        bugnumber="llvm.org/pr21765")
    @skipIfReproducer # Timeouts are not currently modeled.
    def test(self):
        """Test calling std::String member function."""
        self.build()

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, 'stop here in main.', self.main_source_spec)

        # First set the timeout too short, and make sure we fail.
        options = lldb.SBExpressionOptions()
        options.SetTimeoutInMicroSeconds(10)
        options.SetUnwindOnError(True)

        frame = thread.GetFrameAtIndex(0)

        value = frame.EvaluateExpression("wait_a_while(1000000)", options)
        self.assertTrue(value.IsValid())
        self.assertFalse(value.GetError().Success())

        # Now do the same thing with the command line command, and make sure it
        # works too.
        interp = self.dbg.GetCommandInterpreter()

        result = lldb.SBCommandReturnObject()
        return_value = interp.HandleCommand(
            "expr -t 100 -u true -- wait_a_while(1000000)", result)
        self.assertEquals(return_value, lldb.eReturnStatusFailed)

        # Okay, now do it again with long enough time outs:

        options.SetTimeoutInMicroSeconds(1000000)
        value = frame.EvaluateExpression("wait_a_while (1000)", options)
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())

        # Now do the same thingwith the command line command, and make sure it
        # works too.
        interp = self.dbg.GetCommandInterpreter()

        result = lldb.SBCommandReturnObject()
        return_value = interp.HandleCommand(
            "expr -t 1000000 -u true -- wait_a_while(1000)", result)
        self.assertEquals(return_value, lldb.eReturnStatusSuccessFinishResult)

        # Finally set the one thread timeout and make sure that doesn't change
        # things much:

        options.SetTimeoutInMicroSeconds(1000000)
        options.SetOneThreadTimeoutInMicroSeconds(500000)
        value = frame.EvaluateExpression("wait_a_while (1000)", options)
        self.assertTrue(value.IsValid())
        self.assertTrue(value.GetError().Success())
