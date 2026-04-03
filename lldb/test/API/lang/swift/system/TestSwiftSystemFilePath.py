"""
Test System.FilePath and SystemString summary strings.
"""

import lldb

from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil

class TestSwiftSystemFilePath(TestBase):
    @swiftTest
    @skipUnlessDarwin
    def test(self):
        """Test that System.FilePath and SystemString are formatted correctly."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        # Test FilePath summaries.
        path = self.frame().FindVariable("path")
        self.assertEqual(path.GetSummary(), '"/usr/local/bin"')

        empty = self.frame().FindVariable("empty")
        self.assertEqual(empty.GetSummary(), '""')

        root = self.frame().FindVariable("root")
        self.assertEqual(root.GetSummary(), '"/"')

        nonASCII = self.frame().FindVariable("nonASCII")
        self.assertEqual(nonASCII.GetSummary(), '"/hëllo/wôrld"')

        # Test SystemString summary via the _storage child of a FilePath.
        # These checks are guarded because _storage is an internal
        # implementation detail that may change in future Swift versions.
        storage = path.GetChildMemberWithName("_storage")
        if storage.IsValid():
            self.assertEqual(
                storage.GetSummary(),
                "['/', 'u', 's', 'r', '/', 'l', 'o', 'c', 'a', 'l', '/', 'b', 'i', 'n', 0x00]",
            )

            # For an ASCII-only path, children should not be printed.
            self.expect(
                "frame variable path",
                matching=False,
                substrs=["_storage ="],
            )

            # Test SystemString summary for a non-ASCII path.
            nonASCII_storage = nonASCII.GetChildMemberWithName("_storage")
            self.assertTrue(nonASCII_storage.IsValid())
            self.assertEqual(
                nonASCII_storage.GetSummary(),
                "['/', 'h', 0xC3, 0xAB, 'l', 'l', 'o', '/', 'w', 0xC3, 0xB4, 'r', 'l', 'd', 0x00]",
            )

            # For a non-ASCII path, children (the _storage byte array)
            # should be printed alongside the summary.
            self.expect(
                "frame variable nonASCII",
                substrs=["_storage ="],
            )
