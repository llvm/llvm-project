"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_launch_environment_with_object(DAPTestCaseBase):
    """
    Tests launch of a simple program with environment variables
    """

    @skipIfWindows
    def test(self):
        program = self.getBuildArtifact("a.out")
        expected_env = {
            "NO_VALUE": None,
            "WITH_VALUE": "BAR",
            "EMPTY_VALUE": "",
            "SPACE": "Hello World",
        }

        session = self.build_and_create_session()
        process_event = session.launch(LaunchArgs(program=program, env=expected_env))
        session.verify_process_exited(after=process_event)

        # Now get the STDOUT and verify our arguments got passed correctly.
        output = session.get_stdout()
        self.assertTrue(output, "expect program output")

        # Collect environment lines.
        env_output = "\n".join(l for l in output.splitlines() if l.startswith("env["))
        # Make sure each environment variable in "expected_env" is actually set in the
        # program environment and contains the right value.
        for variable, value in expected_env.items():
            expected_value = value or ""
            expected_str = f'"{variable}={expected_value}"'
            self.assertIn(
                expected_str,
                env_output,
                f"\n{expected_str} must exist in program's environment \n{env_output}",
            )
