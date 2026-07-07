"""
Test lldb-dap launch request.
"""

import os

from lldbsuite.test.tools.lldb_dap.dap_types import LaunchArgs
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase


class TestDAP_launch_cwd(DAPTestCaseBase):
    """
    Tests the default launch of a simple program with a current working
    directory.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        program_parent_dir = os.path.realpath(os.path.dirname(os.path.dirname(program)))
        session = self.build_and_create_session()
        session.launch(LaunchArgs(program=program, cwd=program_parent_dir))
        session.verify_process_exited()

        # Now get the STDOUT and verify our program's working directory is correct
        output = session.get_stdout()
        self.assertTrue(output and len(output) > 0, "expect program output")

        lines = output.splitlines()
        cwd_lines = [line for line in lines if line.startswith('cwd = "')]
        self.assertEqual(len(cwd_lines), 1, "verified program working directory")
        cwd_line = cwd_lines[0]

        quote_path = f'"{program_parent_dir}"'
        self.assertIn(
            quote_path,
            cwd_line,
            f"working directory '{program_parent_dir}' not in '{cwd_line}'",
        )
