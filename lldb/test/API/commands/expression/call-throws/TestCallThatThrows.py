"""
Test calling a function that throws an ObjC exception, make sure that it doesn't propagate the exception.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCommandWithThrowTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.main_source = "call-throws.m"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    @add_test_categories(["objc"])
    def test(self):
        """Test calling a function that throws and ObjC exception."""
        self.build()
        self.call_function()

    def check_after_call(self):
        # Check that we are back where we were before:
        frame = self.thread.GetFrameAtIndex(0)
        self.assertEqual(
            self.orig_frame_pc, frame.GetPC(), "Restored the zeroth frame correctly"
        )

    def call_function(self):
        """Test calling function that throws."""
        (target, process, self.thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "I am about to throw.", self.main_source_spec
        )

        options = lldb.SBExpressionOptions()
        options.SetUnwindOnError(True)

        frame = self.thread.GetFrameAtIndex(0)
        # Store away the PC to check that the functions unwind to the right
        # place after calls
        self.orig_frame_pc = frame.GetPC()

        value = frame.EvaluateExpression("[my_class callMeIThrow]", options)
        self.assertTrue(value.IsValid())
        self.assertFalse(value.GetError().Success())

        self.check_after_call()

        # Okay, now try with a breakpoint in the called code in the case where
        # we are ignoring breakpoint hits.
        handler_bkpt = target.BreakpointCreateBySourceRegex(
            "I felt like it", self.main_source_spec
        )
        self.assertGreater(handler_bkpt.GetNumLocations(), 0)
        options.SetIgnoreBreakpoints(True)
        options.SetUnwindOnError(True)

        value = frame.EvaluateExpression("[my_class callMeIThrow]", options)

        self.assertTrue(value.IsValid() and value.GetError().Success() == False)
        self.check_after_call()

        # Now set the ObjC language breakpoint and make sure that doesn't
        # interfere with the call:
        exception_bkpt = target.BreakpointCreateForException(
            lldb.eLanguageTypeObjC, False, True
        )
        self.assertGreater(exception_bkpt.GetNumLocations(), 0)

        options.SetIgnoreBreakpoints(True)
        options.SetUnwindOnError(True)

        value = frame.EvaluateExpression("[my_class callMeIThrow]", options)

        self.assertTrue(value.IsValid() and value.GetError().Success() == False)
        self.check_after_call()

        # Now turn off exception trapping, and call a function that catches the exceptions,
        # and make sure the function actually completes, and we get the right
        # value:
        options.SetTrapExceptions(False)
        value = frame.EvaluateExpression("[my_class iCatchMyself]", options)
        self.assertTrue(value.IsValid())
        self.assertSuccess(value.GetError())
        self.assertEqual(value.GetValueAsUnsigned(), 57)
        self.check_after_call()
        options.SetTrapExceptions(True)

        # Now set this unwind on error to false, and make sure that we stop
        # where the exception was thrown
        options.SetUnwindOnError(False)
        value = frame.EvaluateExpression("[my_class callMeIThrow]", options)

        self.assertTrue(value.IsValid() and value.GetError().Success() == False)
        self.check_after_call()
