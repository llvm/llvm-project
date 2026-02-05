"""
Test that raw value enums are properly resolved in embedded Swift.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedRawValueEnum(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Test that raw value enums display correctly in embedded Swift."""
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        # Test Int raw value enum
        self.expect(
            "frame variable intZero",
            substrs=["IntRawEnum", "zero"],
        )
        self.expect(
            "frame variable intOne",
            substrs=["IntRawEnum", "one"],
        )
        self.expect(
            "frame variable intHundred",
            substrs=["IntRawEnum", "hundred"],
        )

        # Test Int8 raw value enum
        self.expect(
            "frame variable int8A",
            substrs=["Int8RawEnum", "a"],
        )
        self.expect(
            "frame variable int8B",
            substrs=["Int8RawEnum", "b"],
        )

        # Test UInt raw value enum
        self.expect(
            "frame variable uintX",
            substrs=["UIntRawEnum", "x"],
        )
        self.expect(
            "frame variable uintY",
            substrs=["UIntRawEnum", "y"],
        )

        # Test UInt8 raw value enum (C-style)
        self.expect(
            "frame variable uint8First",
            substrs=["UInt8RawEnum", "first"],
        )
        self.expect(
            "frame variable uint8Second",
            substrs=["UInt8RawEnum", "second"],
        )

        # Test Int16 raw value enum with negative values
        self.expect(
            "frame variable int16Neg",
            substrs=["Int16RawEnum", "neg"],
        )
        self.expect(
            "frame variable int16Zero",
            substrs=["Int16RawEnum", "zero"],
        )
        self.expect(
            "frame variable int16Pos",
            substrs=["Int16RawEnum", "pos"],
        )
