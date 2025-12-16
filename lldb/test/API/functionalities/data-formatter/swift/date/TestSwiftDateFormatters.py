"""
Test Foundation.Date summary strings.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

import sys


class TestCase(TestBase):
    @skipUnlessFoundation
    @swiftTest
    def test_swift_date_formatters(self):
        """Test Date summary strings."""
        self.build()

        _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame var date",
            startstr="(Foundation.Date) date = 2001-01-15 13:12:00 UTC",
        )

        if sys.platform != "win32":
            return

        self.expect(
            "frame var nsdate",
            startstr="(Foundation.NSDate) date = 2001-01-15 13:12:00 UTC",
        )
