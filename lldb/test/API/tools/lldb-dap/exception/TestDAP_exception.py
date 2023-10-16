"""
Test exception behavior in DAP
"""


from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase


class TestDAP_exception(lldbdap_testcase.DAPTestCaseBase):
    @skipIfWindows
    def test_stopped_description(self):
        """
        Test that exception description is shown correctly in stopped
        event.
        """
        program = self.getBuildArtifact("a.out")
        print("test_stopped_description called", flush=True)
        self.build_and_launch(program)

        self.dap.request_continue()
        self.assertTrue(self.verify_stop_exception_info("signal SIGABRT"))
