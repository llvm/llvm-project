"""
Test exception behavior in DAP with c++ throw.
"""


from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_exception_cpp(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    def test_stopped_description(self):
        """
        Test that exception description is shown correctly in stopped
        event.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        self.dap_server.request_continue()
        self.assertTrue(self.verify_stop_exception_info("signal SIGABRT"))
        exceptionInfo = self.get_exceptionInfo()
        self.assertEqual(exceptionInfo["breakMode"], "always")
        self.assertEqual(exceptionInfo["description"], "signal SIGABRT")
        self.assertEqual(exceptionInfo["exceptionId"], "signal")
        self.assertIsNotNone(exceptionInfo["details"])
