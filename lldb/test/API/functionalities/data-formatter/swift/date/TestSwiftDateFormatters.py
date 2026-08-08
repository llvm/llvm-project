"""
Test Foundation.Date summary strings.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

import sys


class TestCase(TestBase):
    @requireNotEmbeddedSwift
    @skipUnlessFoundationEssentials
    @skipIfLinux  # https://github.com/swiftlang/llvm-project/issues/13465
    @swiftTest
    def test_swift_date_formatters(self):
        """Test Date summary strings."""
        self.build()

        _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame var date",
            patterns=[
                r"\(Foundation(Essentials)?\.Date\) date = 2001-01-15 13:12:00 UTC"
            ],
        )

        if sys.platform != "win32":
            return

        self.expect(
            "frame var nsdate",
            substrs=["(Foundation.NSDate) nsdate = ", "2001-01-15 13:12:00 UTC"],
        )
