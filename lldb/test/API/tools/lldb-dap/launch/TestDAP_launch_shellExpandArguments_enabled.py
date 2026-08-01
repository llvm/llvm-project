"""
Test lldb-dap launch request.
"""

import os

from lldbsuite.test.decorators import expectedFailureAll, skipIfLinux
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_launch_shellExpandArguments_enabled(DAPTestCaseBase):
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
        session = self.build_and_create_session()
        process_event = session.launch(
            LaunchArgs(program=program, args=[glob], shellExpandArguments=True)
        )
        session.verify_process_exited(after=process_event)

        # Now get the STDOUT and verify our program argument is correct
        output = session.get_stdout()
        self.assertTrue(output and len(output) > 0, "expect program output")
        for line in output.splitlines():
            if line.startswith("arg[1] ="):
                quote_path = f'"{program}"'
                self.assertIn(
                    quote_path,
                    line,
                    f'verify "{glob}" expanded to "{program}"',
                )
