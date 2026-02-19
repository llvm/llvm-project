"""
Test for lldb disassemble command with -Y option and invalid parameters.
This test verifies that disassemble -Y command properly reports error messages
when invoked with incorrect options.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil


class TestDisassembleInvalidTargetFeatures(TestBase):
    """Test disassemble -Y option error handling."""

    def run_invalid_disasm_cmd(self, option, expected_error):
        cmd = f"disassemble -Y {option}"
        self.runCmd(cmd, check=False)
        output = self.res.GetOutput()
        error = self.res.GetError()
        self.assertFalse(self.res.Succeeded(), f"{cmd} should fail")
        self.assertTrue(len(error) > 0, f"Error for '{cmd}' should not be empty")
        self.assertIn(expected_error, error)

    def test_disassemble_Y_invalid_options(self):
        self.build()
        _, _, _, _ = lldbutil.run_to_source_breakpoint(
            self, "main", lldb.SBFileSpec("main.cpp")
        )

        self.run_invalid_disasm_cmd("", "last option requires an argument")
        self.run_invalid_disasm_cmd(
            "invalid", "Unable to find Disassembler plug-in with such parameters:"
        )
        self.run_invalid_disasm_cmd(
            "+invalid_1,-invalid_2,+invalid3", "Failed to disassemble memory at"
        )
        self.run_invalid_disasm_cmd("-Z", "Failed to disassemble memory at")
        self.run_invalid_disasm_cmd("+++", "Failed to disassemble memory at")
        self.run_invalid_disasm_cmd("----", "Failed to disassemble memory at")
