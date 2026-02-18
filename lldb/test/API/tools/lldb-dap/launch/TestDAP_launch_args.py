"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import expectedFailureWindows
import lldbdap_testcase


class TestDAP_launch_args(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests launch of a simple program with arguments
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        args = ["one", "with space", "'with single quotes'", '"with double quotes"']
        self.build_and_launch(program, args=args)
        self.continue_to_exit()

        # Now get the STDOUT and verify our arguments got passed correctly
        output = self.get_stdout()
        self.assertTrue(output and len(output) > 0, "expect program output")
        lines = output.splitlines()
        # Skip the first argument that contains the program name
        lines.pop(0)
        # Make sure arguments we specified are correct
        for i, arg in enumerate(args):
            quoted_arg = '"%s"' % (arg)
            self.assertIn(
                quoted_arg,
                lines[i],
                'arg[%i] "%s" not in "%s"' % (i + 1, quoted_arg, lines[i]),
            )
