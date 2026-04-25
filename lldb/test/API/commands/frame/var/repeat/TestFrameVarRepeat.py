import lldb
from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        """Test that repeating 'frame variable' increments --depth."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        # Start with --depth 0 showing a, but not b.
        self.expect(
            "frame variable --depth 0 a",
            inHistory=True,
            patterns=[r"\(A\) a = {", r"(?!.*b = {)"],
        )

        # First repeat: --depth 1 showing b, but not c.
        self.expect("", patterns=["b = {", "(?!.*c = {)"])

        # Second repeat: --depth 2, showing c, but not d.
        self.expect("", patterns=["c = {", "(?!.*d = {)"])

        # Third repeat: --depth 3, showing d, but not value.
        self.expect("", patterns=["d = {", "(?!.*value = 42)"])

        # Fourth repeat: --depth 4, showing value, the deepest value.
        self.expect("", substrs=["value = 42"])
