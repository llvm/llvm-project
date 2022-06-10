"""
Test some SBValue APIs.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ChangeValueAPITestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # We'll use the test method name as the exe_name.
        self.exe_name = self.testMethodName
        # Find the line number to of function 'c'.
        self.line = line_number('main.c', '// Stop here and set values')
        self.check_line = line_number(
            'main.c', '// Stop here and check values')
        self.end_line = line_number(
            'main.c', '// Set a breakpoint here at the end')

    @expectedFlakeyLinux("llvm.org/pr25652")
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24772")
    def test_change_value(self):
        """Exercise the SBValue::SetValueFromCString API."""
        d = {'EXE': self.exe_name}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        exe = self.getBuildArtifact(self.exe_name)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint inside function 'main'.
        breakpoint = target.BreakpointCreateByLocation('main.c', self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Create the breakpoint inside the function 'main'
        check_breakpoint = target.BreakpointCreateByLocation(
            'main.c', self.check_line)
        self.assertTrue(check_breakpoint, VALID_BREAKPOINT)

        # Create the breakpoint inside function 'main'.
        end_breakpoint = target.BreakpointCreateByLocation(
            'main.c', self.end_line)
        self.assertTrue(end_breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Get Frame #0.
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.IsValid(), "Got a valid frame.")

        # Get the val variable and change it:
        error = lldb.SBError()

        val_value = frame0.FindVariable("val")
        self.assertTrue(val_value.IsValid(), "Got the SBValue for val")
        actual_value = val_value.GetValueAsSigned(error, 0)
        self.assertSuccess(error, "Got a value from val")
        self.assertEquals(actual_value, 100, "Got the right value from val")

        result = val_value.SetValueFromCString("12345")
        self.assertTrue(result, "Setting val returned True.")
        actual_value = val_value.GetValueAsSigned(error, 0)
        self.assertSuccess(error, "Got a changed value from val")
        self.assertEqual(
            actual_value, 12345,
            "Got the right changed value from val")

        # Now check that we can set a structure element:

        mine_value = frame0.FindVariable("mine")
        self.assertTrue(mine_value.IsValid(), "Got the SBValue for mine")

        mine_second_value = mine_value.GetChildMemberWithName("second_val")
        self.assertTrue(
            mine_second_value.IsValid(),
            "Got second_val from mine")
        actual_value = mine_second_value.GetValueAsUnsigned(error, 0)
        self.assertTrue(
            error.Success(),
            "Got an unsigned value for second_val")
        self.assertEquals(actual_value, 5555)

        result = mine_second_value.SetValueFromCString("98765")
        self.assertTrue(result, "Success setting mine.second_value.")
        actual_value = mine_second_value.GetValueAsSigned(error, 0)
        self.assertTrue(
            error.Success(),
            "Got a changed value from mine.second_val")
        self.assertEquals(actual_value, 98765,
                        "Got the right changed value from mine.second_val")

        # Next do the same thing with the pointer version.
        ptr_value = frame0.FindVariable("ptr")
        self.assertTrue(ptr_value.IsValid(), "Got the SBValue for ptr")

        ptr_second_value = ptr_value.GetChildMemberWithName("second_val")
        self.assertTrue(ptr_second_value.IsValid(), "Got second_val from ptr")
        actual_value = ptr_second_value.GetValueAsUnsigned(error, 0)
        self.assertTrue(
            error.Success(),
            "Got an unsigned value for ptr->second_val")
        self.assertEquals(actual_value, 6666)

        result = ptr_second_value.SetValueFromCString("98765")
        self.assertTrue(result, "Success setting ptr->second_value.")
        actual_value = ptr_second_value.GetValueAsSigned(error, 0)
        self.assertTrue(
            error.Success(),
            "Got a changed value from ptr->second_val")
        self.assertEquals(actual_value, 98765,
                        "Got the right changed value from ptr->second_val")

        # gcc may set multiple locations for breakpoint
        breakpoint.SetEnabled(False)

        # Now continue.
        process.Continue()

        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition")

        expected_value = "Val - 12345 Mine - 55, 98765, 55555555. Ptr - 66, 98765, 66666666"
        stdout = process.GetSTDOUT(1000)
        self.assertTrue(
            expected_value in stdout,
            "STDOUT showed changed values.")

        # Finally, change the stack pointer to 0, and we should not make it to
        # our end breakpoint.
        frame0 = thread.GetFrameAtIndex(0)
        self.assertTrue(frame0.IsValid(), "Second time: got a valid frame.")
        sp_value = frame0.FindValue("sp", lldb.eValueTypeRegister)
        self.assertTrue(sp_value.IsValid(), "Got a stack pointer value")
        result = sp_value.SetValueFromCString("1")
        self.assertTrue(result, "Setting sp returned true.")
        actual_value = sp_value.GetValueAsUnsigned(error, 0)
        self.assertSuccess(error, "Got a changed value for sp")
        self.assertEqual(
            actual_value, 1,
            "Got the right changed value for sp.")

        # Boundary condition test the SBValue.CreateValueFromExpression() API.
        # LLDB should not crash!
        nosuchval = mine_value.CreateValueFromExpression(None, None)

        process.Continue()

        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread is None,
            "We should not have managed to hit our second breakpoint with sp == 1")

        process.Kill()
