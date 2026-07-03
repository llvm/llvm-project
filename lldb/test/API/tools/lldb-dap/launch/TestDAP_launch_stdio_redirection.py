"""
Test lldb-dap launch request.
"""

import tempfile

from lldbsuite.test.tools.lldb_dap.dap_types import LaunchArgs
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase


class TestDAP_launch_stdio_redirection(DAPTestCaseBase):
    """
    Test stdio redirection.
    """

    def test(self):
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()

        with tempfile.NamedTemporaryFile("rt") as f:
            process_event = session.launch(
                LaunchArgs(program=program, stdio=[None, f.name])
            )
            session.verify_process_exited(after=process_event)
            lines = f.readlines()
            self.assertIn(
                program, lines[0], "make sure program path is in first argument"
            )
