"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import skipIfWindows
import lldbdap_testcase


class TestDAP_launch_disableSTDIO(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests the default launch of a simple program with STDIO disabled.
    """

    @skipIfWindows
    def test(self):
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program, disableSTDIO=True)
        self.continue_to_exit()
        # Now get the STDOUT and verify our program argument is correct
        output = self.get_stdout()
        self.assertEqual(output, "", "expect no program output")
