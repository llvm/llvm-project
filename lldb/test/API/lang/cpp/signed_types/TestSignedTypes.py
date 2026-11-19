"""
Test that variables with signed types display correctly.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class SignedTypesTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.source = "main.cpp"
        self.line = line_number(self.source, "// Set break point at this line.")

    @expectedFailureAll(
        oslist=["windows"],
        debug_info=no_match(["pdb"]),
        archs=["i[3-6]86", "aarch64"],
        bugnumber="llvm.org/pr24489",
    )
    def test(self):
        """Test that variables with signed types display correctly."""
        self.build()

        # Run in synchronous mode
        self.dbg.SetAsync(False)

        _, _, thread, bkpt = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec(self.source), self.line
        )
        self.assertEqual(bkpt.GetNumLocations(), 1)
        self.assertEqual(thread.GetFrameAtIndex(0).GetLineEntry().GetLine(), self.line)

        lldbutil.check_breakpoint(self, bpno=1, expected_hit_count=1)

        # Execute the puts().
        self.runCmd("thread step-over")

        # Test that signed types display correctly.
        self.expect(
            "frame variable --show-types --no-args",
            VARIABLES_DISPLAYED_CORRECTLY,
            patterns=[
                r"\((signed char|char)\) the_signed_char = 'c'",
                r"\((short int|short)\) the_signed_short = 99",
            ],
            substrs=[
                "(int) the_signed_int = 99",
                "(long) the_signed_long = 99",
                "(long long) the_signed_long_long = 99",
            ],
        )
