"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import (
    skipIfLinux,
    expectedFailureWindows,
    expectedFailureAll,
)
import lldbdap_testcase
import os


class TestDAP_launch_shellExpandArguments_enabled(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests the default launch of a simple program with shell expansion
    enabled.
    """

    @skipIfLinux  # shell argument expansion doesn't seem to work on Linux
    @expectedFailureAll(
        oslist=["freebsd", "netbsd", "windows"], bugnumber="llvm.org/pr48349"
    )
    def test(self):
        program = self.getBuildArtifact("a.out")
        program_dir = os.path.dirname(program)
        glob = os.path.join(program_dir, "*.out")
        self.build_and_launch(program, args=[glob], shellExpandArguments=True)
        self.continue_to_exit()
        # Now get the STDOUT and verify our program argument is correct
        output = self.get_stdout()
        self.assertTrue(output and len(output) > 0, "expect no program output")
        lines = output.splitlines()
        for line in lines:
            quote_path = '"%s"' % (program)
            if line.startswith("arg[1] ="):
                self.assertIn(
                    quote_path, line, 'verify "%s" expanded to "%s"' % (glob, program)
                )
