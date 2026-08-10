"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_launch_disableSTDIO(DAPTestCaseBase):
    """
    Tests the default launch of a simple program with STDIO disabled.
    """

    @skipIfWindows
    def test(self):
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        session.launch(LaunchArgs(program=program, disableSTDIO=True))
        session.verify_process_exited()

        # Now get the STDOUT and verify our program argument is correct.
        output = session.get_stdout()
        self.assertEqual(output, "", "expect no program output")
