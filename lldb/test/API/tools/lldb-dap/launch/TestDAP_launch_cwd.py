"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import skipIfWindows
import lldbdap_testcase
import os


class TestDAP_launch_cwd(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests the default launch of a simple program with a current working
    directory.
    """

    @skipIfWindows
    def test(self):
        program = self.getBuildArtifact("a.out")
        program_parent_dir = os.path.realpath(os.path.dirname(os.path.dirname(program)))
        self.build_and_launch(program, cwd=program_parent_dir)
        self.continue_to_exit()
        # Now get the STDOUT and verify our program argument is correct
        output = self.get_stdout()
        self.assertTrue(output and len(output) > 0, "expect program output")
        lines = output.splitlines()
        found = False
        for line in lines:
            if line.startswith('cwd = "'):
                quote_path = '"%s"' % (program_parent_dir)
                found = True
                self.assertIn(
                    quote_path,
                    line,
                    "working directory '%s' not in '%s'" % (program_parent_dir, line),
                )
        self.assertTrue(found, "verified program working directory")
