"""
Test String.Index summary strings.

The test cases are LLDB versions of the original Swift tests, see
https://github.com/apple/swift/pull/58479
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @skipUnlessFoundation
    @swiftTest
    def test_swift_string_index_formatters(self):
        """Test String.Index summary strings."""
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        #
        # The first breakpoint stop tests a native (non-bridged) String.
        #

        self.expect(
            "v nativeIndices",
            substrs=[
                "0[any]",
                "1[utf8]",
                "9[utf8]",
                "10[utf8]",
            ],
        )

        self.expect(
            "v unicodeScalarIndices",
            substrs=[
                "0[any]",
                "1[utf8]",
                "5[utf8]",
                "9[utf8]",
                "10[utf8]",
            ],
        )

        self.expect(
            "v utf8Indices",
            substrs=[
                "0[any]",
                "1[utf8]",
                "2[utf8]",
                "3[utf8]",
                "4[utf8]",
                "5[utf8]",
                "6[utf8]",
                "7[utf8]",
                "8[utf8]",
                "9[utf8]",
                "10[utf8]",
            ],
        )

        self.expect(
            "v utf16Indices",
            substrs=[
                "0[any]",
                "1[utf8]",
                "1[utf8]+1",
                "5[utf8]",
                "5[utf8]+1",
                "9[utf8]",
                "10[utf8]",
            ],
        )

        #
        # The second breakpoint stop tests a bridged String.
        #

        process.Continue()

        self.expect(
            "v nativeIndices",
            substrs=[
                "0[any]",
                "1[utf16]",
                "5[utf16]",
                "6[utf16]",
            ],
        )

        self.expect(
            "v unicodeScalarIndices",
            substrs=[
                "0[any]",
                "1[utf16]",
                "3[utf16]",
                "5[utf16]",
                "6[utf16]",
            ],
        )

        self.expect(
            "v utf8Indices",
            substrs=[
                "0[any]",
                "1[utf16]",
                "1[utf16]+1",
                "1[utf16]+2",
                "1[utf16]+3",
                "3[utf16]",
                "3[utf16]+1",
                "3[utf16]+2",
                "3[utf16]+3",
                "5[utf16]",
                "6[utf16]",
            ],
        )

        self.expect(
            "v utf16Indices",
            substrs=[
                "0[any]",
                "1[utf16]",
                "2[utf16]",
                "3[utf16]",
                "4[utf16]",
                "5[utf16]",
                "6[utf16]",
            ],
        )
