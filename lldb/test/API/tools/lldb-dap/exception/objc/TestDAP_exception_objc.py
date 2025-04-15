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
