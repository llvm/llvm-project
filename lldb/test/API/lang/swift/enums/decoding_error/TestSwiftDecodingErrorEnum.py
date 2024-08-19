import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os


class TestCase(TestBase):
    @swiftTest
    @skipUnlessFoundation
    def test_swift_decoding_error(self):
        """Regression test for Swift.DecodingError, a specific instance of a multipayload enum."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        substrs = [
            "(DecodingError) ", " = keyNotFound {",
            'debugDescription = "No value associated with key CodingKeys(stringValue: \\"number\\", intValue: nil)',
        ]
        self.expect("frame variable error", substrs=substrs)
        self.expect("expression error", substrs=substrs)
