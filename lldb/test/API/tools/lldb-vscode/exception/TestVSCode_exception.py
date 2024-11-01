"""
Test exception behavior in VSCode
"""


from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbvscode_testcase


class TestVSCode_exception(lldbvscode_testcase.VSCodeTestCaseBase):
    @skipIfWindows
    def test_stopped_description(self):
        """
        Test that exception description is shown correctly in stopped
        event.
        """
        program = self.getBuildArtifact("a.out")
        print("test_stopped_description called", flush=True)
        self.build_and_launch(program)

        self.vscode.request_continue()
        self.assertTrue(self.verify_stop_exception_info("signal SIGABRT"))
