import lldb
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    def test_explicit_depth(self):
        """Test that repeating 'frame variable' increments --depth."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        # Start with --depth 2 showing a, b, and c, but but not d.
        self.expect(
            "frame variable --depth 2 a",
            inHistory=True,
            patterns=[r"\(A\) a = {", "b = {", "c = {", r"(?!.*d = {)"],
        )

        # First repeat: --depth 4, showing d, but not e.
        self.expect("", patterns=["d = {", "(?!.*e = {)"])

        # Second repeat: --depth 5, showing e, but not f.
        self.expect("", patterns=["e = {", "(?!.*f = {)"])

        # Third repeat: --depth 6, showing d, but not f.
        self.expect("", patterns=["e = {", "(?!.*leaf = 42)"])

        # Fourth repeat: --depth 7, showing leaf, the deepest child.
        self.expect("", substrs=["leaf = 42"])

    def test_default_depth(self):
        """Test that repeating 'frame variable' adds a --depth option."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        # Default depth shows f but not leaf.
        self.expect(
            "frame variable a",
            inHistory=True,
            patterns=[r"f = \{...\}", r"(?!.*leaf = 42)"],
        )

        # Repeat
        self.expect("", substrs=["leaf = 42"])
