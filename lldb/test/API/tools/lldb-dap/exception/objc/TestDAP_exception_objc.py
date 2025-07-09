"""
Test exception behavior in DAP with obj-c throw.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_exception_objc(lldbdap_testcase.DAPTestCaseBase):
    @skipUnlessDarwin
    def test_stopped_description(self):
        """
        Test that exception description is shown correctly in stopped event.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        self.dap_server.request_continue()
        self.assertTrue(self.verify_stop_exception_info("signal SIGABRT"))
        exception_info = self.get_exceptionInfo()
        self.assertEqual(exception_info["breakMode"], "always")
        self.assertEqual(exception_info["description"], "signal SIGABRT")
        self.assertEqual(exception_info["exceptionId"], "signal")
        exception_details = exception_info["details"]
        self.assertRegex(exception_details["message"], "SomeReason")
        self.assertRegex(exception_details["stackTrace"], "main.m")

    @skipUnlessDarwin
    def test_break_on_throw_and_catch(self):
        """
        Test that breakpoints on exceptions work as expected.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        response = self.dap_server.request_setExceptionBreakpoints(
            filter_options=[
                {
                    "filterId": "objc_throw",
                    "condition": '[[((NSException *)$arg1) name] isEqual:@"ThrownException"]',
                },
            ]
        )
        if response:
            self.assertTrue(response["success"])

        self.continue_to_exception_breakpoint("Objective-C Throw")

        # FIXME: Catching objc exceptions do not appear to be working.
        # Xcode appears to set a breakpoint on '__cxa_begin_catch' for objc
        # catch, which is different than
        # SBTarget::BreakpointCreateForException(eLanguageObjectiveC, /*catch_bp=*/true, /*throw_bp=*/false);
        # self.continue_to_exception_breakpoint("Objective-C Catch")

        self.do_continue()

        self.assertTrue(self.verify_stop_exception_info("signal SIGABRT"))
        exception_info = self.get_exceptionInfo()
        self.assertEqual(exception_info["breakMode"], "always")
        self.assertEqual(exception_info["description"], "signal SIGABRT")
        self.assertEqual(exception_info["exceptionId"], "signal")
        exception_details = exception_info["details"]
        self.assertRegex(exception_details["message"], "SomeReason")
        self.assertRegex(exception_details["stackTrace"], "main.m")
