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
        self.do_test("--depth ")
        self.do_test("--de ")
        self.do_test("-D ")
        self.do_test("-D")

    def do_test(self, depth_option):
        # Start with --depth 2, shows a, b, and c, but but not d.
        self.expect(
            f"frame variable {depth_option}2 a",
            inHistory=True,
            patterns=[r"\(A\) a = {", "b = {", "c = {", r"(?!.*d = {)"],
        )

        # First repeat: shows d, but not e.
        self.expect("", patterns=["d = {", "(?!.*e = {)"])

        # Second repeat: shows e, but not f.
        self.expect("", patterns=["e = {", "(?!.*f = {)"])

        # Third repeat: shows f, but not leaf.
        self.expect("", patterns=["f = {", "(?!.*leaf = 42)"])

        # Fourth repeat: shows leaf, the deepest child.
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

        # Repeat to show leaf.
        self.expect("", substrs=["leaf = 42"])
