"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxChronoDataFormatterTestCase(TestBase):
    @add_test_categories(["libc++"])
    @skipIf(compiler="clang", compiler_version=["<", "11.0"])
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp", False)
        )

        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect("frame variable ns", substrs=["ns = 1 ns"])
        self.expect("frame variable us", substrs=["us = 12 µs"])
        self.expect("frame variable ms", substrs=["ms = 123 ms"])
        self.expect("frame variable s", substrs=["s = 1234 s"])
        self.expect("frame variable min", substrs=["min = 12345 min"])
        self.expect("frame variable h", substrs=["h = 123456 h"])

        self.expect("frame variable d", substrs=["d = 654321 days"])
        self.expect("frame variable w", substrs=["w = 54321 weeks"])
        self.expect("frame variable m", substrs=["m = 4321 months"])
        self.expect("frame variable y", substrs=["y = 321 years"])

