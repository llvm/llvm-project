"""
Test exception behavior in DAP with obj-c throw.
"""

from lldbsuite.test.decorators import skipUnlessDarwin
from lldbsuite.test.tools.lldb_dap.dap_types import ExceptionFilterOptions, LaunchArgs
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase


class TestDAP_exception_objc(DAPTestCaseBase):
    @skipUnlessDarwin
    def test_stopped_description(self):
        """
        Test that exception description is shown correctly in stopped event.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        process_event = session.launch(LaunchArgs(program))
        stop_event = session.verify_stopped_on_exception(
            after=process_event, expected_description="signal SIGABRT"
        )

        thread_id = self.expect_not_none(stop_event.body.threadId)
        exception_info = session.get_exception_info(thread_id)

        self.assertEqual(exception_info.breakMode, "always")
        exception_description = self.expect_not_none(exception_info.description)
        self.assertIn("signal SIGABRT", exception_description)
        self.assertEqual(exception_info.exceptionId, "signal")

        exception_details = self.expect_not_none(exception_info.details)
        exception_message = self.expect_not_none(exception_details.message)
        self.assertRegex(exception_message, "SomeReason")
        stack_trace = self.expect_not_none(exception_details.stackTrace)
        self.assertRegex(stack_trace, "main.m")

    @skipUnlessDarwin
    def test_break_on_throw_and_catch(self):
        """
        Test that breakpoints on exceptions work as expected.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        with session.configure(LaunchArgs(program)) as ctx:
            response = session.set_exception_breakpoints(
                filters=["objc_catch", "objc_throw"],
                filterOptions=[
                    ExceptionFilterOptions(
                        filterId="objc_throw",
                        condition='[[((NSException *)$arg1) name] isEqual:@"ThrownException"]',
                    )
                ],
            )
            self.assertTrue(response.success)

        session.verify_stopped_on_exception(
            after=ctx.process_event,
            expected_description="hit Objective-C exception",
            expected_text="Objective-C Throw",
        )

        # FIXME: Catching objc exceptions do not appear to be working.
        # Xcode appears to set a breakpoint on '__cxa_begin_catch' for objc
        # catch, which is different than
        # SBTarget::BreakpointCreateForException(eLanguageObjectiveC, /*catch_bp=*/true, /*throw_bp=*/false);
        # self.continue_to_exception_breakpoint("Objective-C Catch")

        stop_event = session.continue_to_exception_breakpoint(
            expected_description="signal SIGABRT"
        )

        thread_id = self.expect_not_none(stop_event.body.threadId)
        exception_info = session.get_exception_info(thread_id)

        self.assertEqual(exception_info.breakMode, "always")
        description = self.expect_not_none(exception_info.description)
        self.assertIn("signal SIGABRT", description)
        self.assertEqual(exception_info.exceptionId, "signal")

        exception_details = self.expect_not_none(exception_info.details)
        message = self.expect_not_none(exception_details.message)
        self.assertRegex(message, "SomeReason")
        stack_trace = self.expect_not_none(exception_details.stackTrace)
        self.assertRegex(stack_trace, "main.m")
