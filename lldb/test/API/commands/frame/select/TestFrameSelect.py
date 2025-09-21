"""
Test 'frame select' command.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestFrameSelect(TestBase):
    @no_debug_info_test
    @skipIfWindows
    def test_relative(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        self.expect("frame select -r 1", substrs=["nested2() at"])
        self.expect("frame select -r -1", substrs=["nested3() at"])

        self.expect(
            "frame select -r -1",
            error=True,
            substrs=["already at the bottom of the stack"],
        )
        self.expect(
            "frame select -r -2147483647",
            error=True,
            substrs=["already at the bottom of the stack"],
        )
        self.expect(
            "frame select -r -2147483648",
            error=True,
            substrs=["error: invalid frame offset argument '-2147483648'"],
        )
        self.expect(
            "frame select -r -2147483649",
            error=True,
            substrs=["error: invalid frame offset argument '-2147483649'"],
        )

        self.expect("frame select -r 1", substrs=["nested2() at"])
        self.expect("frame select -r -2", substrs=["nested3() at"])
        self.expect("frame select -r 1", substrs=["nested2() at"])
        self.expect("frame select -r -2147483647", substrs=["nested3() at"])
        self.expect("frame select -r 1", substrs=["nested2() at"])
        self.expect(
            "frame select -r -2147483648",
            error=True,
            substrs=["error: invalid frame offset argument '-2147483648'"],
        )
        self.expect(
            "frame select -r -2147483649",
            error=True,
            substrs=["error: invalid frame offset argument '-2147483649'"],
        )

        self.expect("frame select -r 100")
        self.expect(
            "frame select -r 1",
            error=True,
            substrs=["already at the top of the stack"],
        )

    @no_debug_info_test
    @skipIfWindows
    def test_mixing_relative_and_abs(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        # The function associated with each frame index can change depending
        # on the function calling main (e.g. `start`), so this only tests that
        # the frame index number is correct. We test the actual functions
        # in the relative test.

        # Jump to the top of the stack.
        self.expect("frame select 0", substrs=["frame #0"])

        # Run some relative commands.
        self.expect("up", substrs=["frame #1"])
        self.expect("frame select -r 1", substrs=["frame #2"])
        self.expect("frame select -r -1", substrs=["frame #1"])

        # Test that absolute indices still work.
        self.expect("frame select 2", substrs=["frame #2"])
        self.expect("frame select 1", substrs=["frame #1"])
        self.expect("frame select 3", substrs=["frame #3"])
        self.expect("frame select 0", substrs=["frame #0"])
        self.expect("frame select 1", substrs=["frame #1"])

        # Run some other relative frame select commands.
        self.expect("down", substrs=["frame #0"])
        self.expect("frame select -r 1", substrs=["frame #1"])
        self.expect("frame select -r -1", substrs=["frame #0"])

        # Test that absolute indices still work.
        self.expect("frame select 2", substrs=["frame #2"])
        self.expect("frame select 1", substrs=["frame #1"])
        self.expect("frame select 3", substrs=["frame #3"])
        self.expect("frame select 0", substrs=["frame #0"])
