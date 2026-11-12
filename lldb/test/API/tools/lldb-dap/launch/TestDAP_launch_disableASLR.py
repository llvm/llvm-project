"""
Test that lldb-dap's launch falls back to target.disable-aslr when the launch
request omits disableASLR.
"""

import re

from lldbsuite.test.decorators import *
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase


class TestDAP_launch_disableASLR(DAPTestCaseBase):
    """
    lldb-dap applies the launch flags it computes to the target before running
    "launchCommands", so `settings show target.disable-aslr` there reports
    whether lldb-dap would disable ASLR. The launch itself always keeps ASLR
    enabled, since containers commonly reject disabling it.
    """

    USE_DEFAULT_DEBUG_ADAPTER = False

    def disable_aslr_after_launch(self, setting_command, disable_aslr):
        session = self.create_session(adapter=self.create_debug_adapter())
        program = self.getBuildArtifact("a.out")
        session.launch(
            LaunchArgs(
                program,
                disableASLR=disable_aslr,
                initCommands=[setting_command],
                launchCommands=[
                    "settings show target.disable-aslr",
                    "process launch --stop-at-entry --disable-aslr false",
                ],
            )
        )
        session.verify_process_exited()
        match = re.search(
            r"target\.disable-aslr \(boolean\) = (true|false)", session.get_console()
        )
        self.assertIsNotNone(match, "expected `settings show` output")
        return match.group(1) == "true"

    def test(self):
        self.build()
        cases = [
            ("settings clear target.disable-aslr", None, True),
            ("settings set target.disable-aslr false", None, False),
            ("settings set target.disable-aslr false", True, True),
            ("settings set target.disable-aslr true", False, False),
        ]
        for setting_command, disable_aslr, expected in cases:
            with self.subTest(setting=setting_command, disableASLR=disable_aslr):
                self.assertEqual(
                    self.disable_aslr_after_launch(setting_command, disable_aslr),
                    expected,
                )
