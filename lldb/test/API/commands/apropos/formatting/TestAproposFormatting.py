"""
Test the formatting and alignment of the apropos command output.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


class AproposFormattingTest(PExpectTest):
    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine.
    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_apropos_with_settings_alignment(self):
        """Test that apropos aligns both commands and settings to the same column."""
        self.launch()
        self.child.setwinsize(24, 120)

        self.child.sendline("apropos disass")

        self.child.expect_exact("The following commands may relate to 'disass':")
        self.child.expect_exact(
            "  disassemble -- Disassemble specified instructions in the current target."
        )

        self.child.expect_exact(
            "The following settings variables may relate to 'disass':"
        )
        self.child.expect_exact(
            "  disassembly-format            -- The default disassembly format string to use"
        )

        self.expect_prompt()
        self.quit()
