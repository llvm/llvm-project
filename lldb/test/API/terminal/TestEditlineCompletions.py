import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbpexpect import PExpectTest


class EditlineCompletionsTest(PExpectTest):
    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_completion_truncated(self):
        """Test that the completion is correctly truncated."""
        self.launch(dimensions=(10, 20))
        self.child.send("_regexp-\t")
        self.child.expect("        _regexp-a...")
        self.child.expect("        _regexp-b...")

    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_description_truncated(self):
        """Test that the description is correctly truncated."""
        self.launch(dimensions=(10, 70))
        self.child.send("_regexp-\t")
        self.child.expect(
            "        _regexp-attach    -- Attach to process by ID or name."
        )
        self.child.expect(
            "        _regexp-break     -- Set a breakpoint using one of several..."
        )

    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_separator_omitted(self):
        """Test that the separated is correctly omitted."""
        self.launch(dimensions=(10, 32))
        self.child.send("_regexp-\t")
        self.child.expect("        _regexp-attach   \r\n")
        self.child.expect("        _regexp-break    \r\n")

    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_separator(self):
        """Test that the separated is correctly printed."""
        self.launch(dimensions=(10, 33))
        self.child.send("_regexp-\t")
        self.child.expect("        _regexp-attach    -- A...")
        self.child.expect("        _regexp-break     -- S...")

    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_multiline_description(self):
        """Test that multi-line descriptions are correctly padded and truncated."""
        self.launch(dimensions=(10, 72))
        self.child.send("k\t")
        self.child.expect(
            "        kdp-remote -- Connect to a process via remote KDP server."
        )
        self.child.expect(
            "                      If no UDP port is specified, port 41139 is assu..."
        )
        self.child.expect(
            "                      kdp-remote is an abbreviation for 'process conn..."
        )
        self.child.expect("        kill       -- Terminate the current target process.")

    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_completion_pagination(self):
        """Test that we use the terminal height for pagination."""
        self.launch(dimensions=(10, 30))
        self.child.send("_regexp-\t")
        self.child.expect("Available completions:")
        self.child.expect("        _regexp-attach")
        self.child.expect("        _regexp-break")
        self.child.expect("        _regexp-bt")
        self.child.expect("        _regexp-display")
        self.child.expect("        _regexp-down")
        self.child.expect("        _regexp-env")
        self.child.expect("        _regexp-jump")
        self.child.expect("More")

    @skipIfAsan
    @skipIfEditlineSupportMissing
    def test_completion_multiline_pagination(self):
        """Test that we use the terminal height for pagination and account for multi-line descriptions."""
        self.launch(dimensions=(6, 72))
        self.child.send("k\t")
        self.child.expect("Available completions:")
        self.child.expect(
            "        kdp-remote -- Connect to a process via remote KDP server."
        )
        self.child.expect(
            "                      If no UDP port is specified, port 41139 is assu..."
        )
        self.child.expect(
            "                      kdp-remote is an abbreviation for 'process conn..."
        )
        self.child.expect("More")
