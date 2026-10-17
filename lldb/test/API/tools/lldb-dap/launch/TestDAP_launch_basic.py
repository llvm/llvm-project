"""
Test lldb-dap launch request.
"""

from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs


class TestDAP_launch_basic(DAPTestCaseBase):
    """
    Tests the default launch of a simple program. No arguments,
    environment, or anything else is specified.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        session.launch(LaunchArgs(program=program))
        session.verify_process_exited()

        # Now get the STDOUT and verify our program argument is correct.
        output = session.get_stdout()
        self.assertTrue(output and len(output) > 0, "expect program output")
        lines = output.splitlines()
        self.assertIn(program, lines[0], "make sure program path is in first argument")
