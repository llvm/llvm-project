import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @swiftTest
    def test(self):
        """Test summary formatter for TaskPriority."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect(
            "frame var",
            substrs=[
                "high = .high",
                "medium = .medium",
                "low = .low",
                "background = .background",
                "custom = 15",
            ],
        )
