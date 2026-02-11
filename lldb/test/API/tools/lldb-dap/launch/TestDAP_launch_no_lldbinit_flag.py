"""
Test lldb-dap launch request.
"""

import lldbdap_testcase
import os
import tempfile


class TestDAP_launch_no_lldbinit_flag(lldbdap_testcase.DAPTestCaseBase):
    """
    Test that the --no-lldbinit flag prevents sourcing .lldbinit files.
    """

    def test(self):
        # Create a temporary .lldbinit file in the home directory
        with tempfile.TemporaryDirectory() as temp_home:
            lldbinit_path = os.path.join(temp_home, ".lldbinit")

            # Write a command to the .lldbinit file that would set a unique setting
            with open(lldbinit_path, "w") as f:
                f.write("settings set stop-disassembly-display never\n")
                f.write("settings set target.x86-disassembly-flavor intel\n")

            # Test with --no-lldbinit flag (should NOT source .lldbinit)
            self.build_and_create_debug_adapter(
                lldbDAPEnv={"HOME": temp_home}, additional_args=["--no-lldbinit"]
            )
            program = self.getBuildArtifact("a.out")

            # Use initCommands to check if .lldbinit was sourced
            initCommands = ["settings show stop-disassembly-display"]

            # Launch with initCommands to check the setting
            self.launch(program, initCommands=initCommands)
            self.continue_to_exit()

            # Get console output to verify the setting was NOT set from .lldbinit
            output = self.get_console()
            self.assertTrue(output and len(output) > 0, "expect console output")

            # Verify the setting has default value, not "never" from .lldbinit
            self.assertNotIn(
                "never",
                output,
                "Setting should have default value when --no-lldbinit is used",
            )

            # Verify the initCommands were executed
            self.verify_commands("initCommands", output, initCommands)
