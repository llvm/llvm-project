"""
Test lldb-dap launch request.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase
import os
import re


class TestDAP_launch_failing_launch_commands(lldbdap_testcase.DAPTestCaseBase):
    """
    Tests "launchCommands" failures prevents a launch.
    """

    def test(self):
        self.build_and_create_debug_adapter()
        program = self.getBuildArtifact("a.out")

        # Run an invalid launch command, in this case a bad path.
        bad_path = os.path.join("bad", "path")
        launchCommands = ['!target create "%s%s"' % (bad_path, program)]

        initCommands = ["target list", "platform list"]
        preRunCommands = ["image list a.out", "image dump sections a.out"]
        response = self.launch_and_configurationDone(
            program,
            initCommands=initCommands,
            preRunCommands=preRunCommands,
            launchCommands=launchCommands,
        )

        self.assertFalse(response["success"])
        self.assertRegex(
            response["body"]["error"]["format"],
            r"Failed to run launch commands\. See the Debug Console for more details",
        )

        # Get output from the console. This should contain both the
        # "initCommands" and the "preRunCommands".
        output = self.get_console()
        # Verify all "initCommands" were found in console output
        self.verify_commands("initCommands", output, initCommands)
        # Verify all "preRunCommands" were found in console output
        self.verify_commands("preRunCommands", output, preRunCommands)

        # Verify all "launchCommands" were founc in console output
        # The launch should fail due to the invalid command.
        self.verify_commands("launchCommands", output, launchCommands)
        self.assertRegex(output, re.escape(bad_path) + r".*does not exist")
