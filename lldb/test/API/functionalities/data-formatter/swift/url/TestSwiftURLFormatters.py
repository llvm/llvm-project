"""
Test Foundation.URL summary strings.
"""

import lldb
import sys

from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @skipUnlessFoundation
    @swiftTest
    def test_swift_url_formatters(self):
        """Test URL summary strings."""
        self.build()

        foundation = "Foundation" if sys.platform == "darwin" else "FoundationEssentials"

        _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame var url",
            startstr=f'({foundation}.URL?) url = "https://www.example.com/path?query#fragment"'
        )

        self.expect(
            "frame var relativeURL",
            startstr=f'({foundation}.URL?) relativeURL = "relative -- https://www.example.com/"'
        )
