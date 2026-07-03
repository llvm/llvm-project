"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import skipIfWindows
import lldbdap_testcase


class TestDAP_launch_environment_with_array(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests launch of a simple program with environment variables
    """

    @skipIfWindows
    def test(self):
        program = self.getBuildArtifact("a.out")
        env = ["NO_VALUE", "WITH_VALUE=BAR", "EMPTY_VALUE=", "SPACE=Hello World"]

        self.build_and_launch(program, env=env)
        self.continue_to_exit()

        # Now get the STDOUT and verify our arguments got passed correctly
        output = self.get_stdout()
        self.assertTrue(output and len(output) > 0, "expect program output")
        lines = output.splitlines()
        # Skip the all arguments so we have only environment vars left
        while len(lines) and lines[0].startswith("arg["):
            lines.pop(0)
        # Make sure each environment variable in "env" is actually set in the
        # program environment that was printed to STDOUT
        for var in env:
            found = False
            for program_var in lines:
                if var in program_var:
                    found = True
                    break
            self.assertTrue(
                found, '"%s" must exist in program environment (%s)' % (var, lines)
            )
