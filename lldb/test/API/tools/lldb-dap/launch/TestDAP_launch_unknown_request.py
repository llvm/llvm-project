"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import expectedFailureWindows
import lldbdap_testcase


class TestDAP_launch_unknown_request(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests handling of unknown request.
    """

    @expectedFailureWindows(
        bugnumber="https://github.com/llvm/llvm-project/issues/137599"
    )
    def test(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)

        response = self.dap_server.request_unknown()
        self.assertFalse(response["success"])
        self.assertEqual(response["body"]["error"]["format"], "Unknown request")

        self.continue_to_exit()
