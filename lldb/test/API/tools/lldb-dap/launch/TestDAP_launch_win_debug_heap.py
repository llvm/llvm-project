"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import skipUnlessWindows
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs, Console
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from typing import List


@skipUnlessWindows
class TestDAP_launch_win_debug_heap(DAPTestCaseBase):
    """
    Test that lldb-dap respects the debug heap setting on Windows when launching in an integrated terminal.
    """

    def run_with(self, *, env: List[str] = [], init_commands: List[str] = []):
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        process_event = session.launch(
            LaunchArgs(
                program=program,
                env=env,
                initCommands=init_commands,
                console=Console.INTEGRATED_TERMINAL,
            )
        )
        session.verify_process_exited(after=process_event)

        output = session.get_stdout()
        self.assertTrue(output, "expect program output")

        return "\n".join(l for l in output.splitlines() if l.startswith("env["))

    def test_default(self):
        env_output = self.run_with()
        self.assertIn("_NO_DEBUG_HEAP=1", env_output)

    def test_default_overwrite(self):
        env_output = self.run_with(env=["_NO_DEBUG_HEAP=2"])
        self.assertIn("_NO_DEBUG_HEAP=2", env_output)

    def test_enabled(self):
        env_output = self.run_with(
            init_commands=[
                "settings set platform.plugin.windows.disable-debug-heap true"
            ],
        )
        self.assertIn("_NO_DEBUG_HEAP=1", env_output)

    def test_disabled(self):
        env_output = self.run_with(
            init_commands=[
                "settings set platform.plugin.windows.disable-debug-heap false"
            ]
        )
        self.assertNotIn("_NO_DEBUG_HEAP", env_output)

    def test_disabled_overwrite(self):
        env_output = self.run_with(
            env=["_NO_DEBUG_HEAP=2"],
            init_commands=[
                "settings set platform.plugin.windows.disable-debug-heap false"
            ],
        )
        self.assertIn("_NO_DEBUG_HEAP=2", env_output)
