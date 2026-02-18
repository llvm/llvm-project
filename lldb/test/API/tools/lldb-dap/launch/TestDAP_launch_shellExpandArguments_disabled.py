"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import expectedFailureWindows
import lldbdap_testcase
import os


class TestDAP_launch_shellExpandArguments_disabled(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests the default launch of a simple program with shell expansion
    disabled.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        program_dir = os.path.dirname(program)
        glob = os.path.join(program_dir, "*.out")
        self.build_and_launch(program, args=[glob], shellExpandArguments=False)
        self.continue_to_exit()
        # Now get the STDOUT and verify our program argument is correct
        output = self.get_stdout()
        self.assertTrue(output and len(output) > 0, "expect no program output")
        lines = output.splitlines()
        for line in lines:
            quote_path = '"%s"' % (glob)
            if line.startswith("arg[1] ="):
                self.assertIn(
                    quote_path, line, 'verify "%s" stayed to "%s"' % (glob, glob)
                )
