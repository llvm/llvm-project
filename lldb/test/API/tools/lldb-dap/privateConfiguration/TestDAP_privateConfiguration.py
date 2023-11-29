"""
Test lldb-dap stack trace response
"""


import os

import dap_server
import lldbdap_testcase
from lldbsuite.test import lldbtest, lldbutil
from lldbsuite.test.decorators import *


class TestDAP_privateConfiguration(lldbdap_testcase.DAPTestCaseBase):
    def do_test_initCommands(self, printMode, includeError=False):
        """
        Test the initCommands property of the privateConfiguration setting and
        its various print modes.
        """
        commands = [
            "settings set target.show-hex-variable-values-with-leading-zeroes false"
        ]
        if includeError:
            commands.append(
                "settings set target.show-hex-variable-values-with-leading-zeroes fooooo"
            )

        program = self.getBuildArtifact("a.out")
        self.build_and_launch(
            program,
            privateConfiguration={
                "initCommands": {
                    "commands": commands,
                    "printMode": printMode,
                }
            },
        )
        full_output = self.collect_console(duration=1.0)
        expected_output = """Running privateInitCommands:
(lldb) settings set target.show-hex-variable-values-with-leading-zeroes false"""
        expected_error_output = "error: invalid boolean string value: 'fooooo'"

        if printMode == "always":
            self.assertIn(expected_output, full_output)
            if includeError:
                self.assertIn(expected_error_output, full_output)
        elif printMode == "never":
            self.assertNotIn(expected_output, full_output)
            if includeError:
                self.assertNotIn(expected_error_output, full_output)
        else:
            if includeError:
                self.assertIn(expected_output, full_output)
                self.assertIn(expected_error_output, full_output)
            else:
                self.assertNotIn(expected_output, full_output)
                self.assertNotIn(expected_error_output, full_output)

    @skipIfWindows
    @skipIfRemote
    def test_initCommands_with_print_mode_always(self):
        self.do_test_initCommands("always")

    @skipIfWindows
    @skipIfRemote
    def test_initCommands_with_print_mode_never(self):
        self.do_test_initCommands("never", includeError=True)

    @skipIfWindows
    @skipIfRemote
    def test_initCommands_with_print_mode_onError_with_failure(self):
        self.do_test_initCommands("onError", includeError=True)

    @skipIfWindows
    @skipIfRemote
    def test_initCommands_with_print_mode_onError_no_actual_failures(self):
        self.do_test_initCommands("onError", includeError=False)
