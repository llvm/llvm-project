"""
Test System.FilePath summary strings.
"""

import lldb

from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil

class TestSwiftSystemFilePath(TestBase):
    @swiftTest
    @skipUnlessDarwin
    def test(self):
        """Test that System.FilePath is formatted correctly."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect(
            "frame var path",
            startstr='(System.FilePath) path = "/usr/local/bin"',
        )
        self.expect(
            "frame var empty",
            startstr='(System.FilePath) empty = ""',
        )
        self.expect(
            "frame var root",
            startstr='(System.FilePath) root = "/"',
        )
