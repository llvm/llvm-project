"""
Test ArraySlice synthetic types.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @swiftTest
    def test_swift_array_slice_formatters(self):
        """Test ArraySlice synthetic types."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        for var in ("someSlice", "arraySlice", "arraySubSequence"):
            self.expect(f"v {var}", substrs=[f"{var} = 2 values", "[1] = 2", "[2] = 3"])
