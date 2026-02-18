"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import (
    skipIfAsan,
    skipIf,
    skipIfBuildType,
    no_match,
    skipIfWindows,
)
import lldbdap_testcase
import tempfile


class TestDAP_launch_stdio_redirection_and_console(lldbdap_testcase.DAPTestCaseBase):
    """
    Test stdio redirection and console.
    """

    @skipIfAsan
    @skipIfWindows  # https://github.com/llvm/llvm-project/issues/62336
    @skipIf(oslist=["linux"], archs=no_match(["x86_64"]))
    @skipIfBuildType(["debug"])
    def test(self):
        self.build_and_create_debug_adapter()
        program = self.getBuildArtifact("a.out")

        with tempfile.NamedTemporaryFile("rt") as f:
            self.launch_and_configurationDone(
                program, console="integratedTerminal", stdio=[None, f.name, None]
            )
            self.verify_process_exited()
            lines = f.readlines()
            self.assertIn(
                program, lines[0], "make sure program path is in first argument"
            )
