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

    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_apropos_highlights_matches(self):
        """Test that apropos highlights matching keywords in output."""
        self.launch(use_colors=True, dimensions=(100, 200))

        ansi_green = "\x1b[32m"
        ansi_reset = "\x1b[0m"
        self.child.sendline(
            "settings set show-regex-match-ansi-prefix ${ansi.fg.green}"
        )
        self.expect_prompt()

        self.child.sendline("apropos disass")
        # Check command name highlighting.
        self.child.expect_exact(ansi_green + "disass" + ansi_reset + "emble")
        # Check settings name highlighting.
        self.child.expect_exact(ansi_green + "disass" + ansi_reset + "embly-format")
        self.expect_prompt()
        self.quit()

    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_apropos_highlights_across_line_break(self):
        """Test that apropos highlights matches that span a line break."""
        self.launch(use_colors=True, dimensions=(24, 68))

        ansi_green = "\x1b[32m"
        ansi_reset = "\x1b[0m"
        self.child.sendline(
            "settings set show-regex-match-ansi-prefix ${ansi.fg.green}"
        )
        self.expect_prompt()

        self.child.sendline('apropos "will override this"')
        # The narrow terminal forces "will override this" to wrap across
        # lines. Verify the ANSI highlight spans the match: green starts
        # before "will" and the reset comes after "this".
        self.child.expect_exact(ansi_green + "will")
        self.child.expect_exact("this" + ansi_reset)
        self.expect_prompt()
        self.quit()
