"""
Test lldb-dap launch request.
"""

import tempfile

from lldbsuite.test.decorators import (
    no_match,
    skipIf,
    skipIfAsan,
    skipIfBuildType,
    skipIfWindows,
)
from lldbsuite.test.tools.lldb_dap.dap_types import Console, LaunchArgs
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase


class TestDAP_launch_stdio_redirection_and_console(DAPTestCaseBase):
    """
    Test stdio redirection and console.
    """

    @skipIfAsan
    @skipIfWindows  # https://github.com/llvm/llvm-project/issues/198763
    @skipIf(oslist=["linux"], archs=no_match(["x86_64"]))
    @skipIfBuildType(["debug"])
    def test(self):
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()

        with tempfile.NamedTemporaryFile("rt") as f:
            process_event = session.launch(
                LaunchArgs(
                    program=program,
                    console=Console.INTEGRATED_TERMINAL,
                    stdio=[None, f.name, None],
                )
            )
            session.verify_process_exited(after=process_event)
            lines = f.readlines()
            self.assertIn(
                program, lines[0], "make sure program path is in first argument"
            )
