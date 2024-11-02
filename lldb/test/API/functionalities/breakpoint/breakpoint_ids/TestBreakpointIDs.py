"""
Test lldb breakpoint ids.
"""



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class BreakpointIDTestCase(TestBase):

    def test(self):
        self.build()

        exe = self.getBuildArtifact("a.out")
        self.expect("file " + exe,
                    patterns=["Current executable set to .*a.out"])

        bpno = lldbutil.run_break_set_by_symbol(
            self, 'product', num_expected_locations=-1, sym_exact=False)
        self.assertEquals(bpno, 1, "First breakpoint number is 1.")

        bpno = lldbutil.run_break_set_by_symbol(
            self, 'sum', num_expected_locations=-1, sym_exact=False)
        self.assertEquals(bpno, 2, "Second breakpoint number is 2.")

        bpno = lldbutil.run_break_set_by_symbol(
            self, 'junk', num_expected_locations=0, sym_exact=False)
        self.assertEquals(bpno, 3, "Third breakpoint number is 3.")

        self.expect(
            "breakpoint disable 1.1 - 2.2 ",
            COMMAND_FAILED_AS_EXPECTED,
            error=True,
            startstr="error: Invalid range: Ranges that specify particular breakpoint locations must be within the same major breakpoint; you specified two different major breakpoints, 1 and 2.")

        self.expect(
            "breakpoint disable 2 - 2.2",
            COMMAND_FAILED_AS_EXPECTED,
            error=True,
            startstr="error: Invalid breakpoint id range:  Either both ends of range must specify a breakpoint location, or neither can specify a breakpoint location.")

        self.expect(
            "breakpoint disable 2.1 - 2",
            COMMAND_FAILED_AS_EXPECTED,
            error=True,
            startstr="error: Invalid breakpoint id range:  Either both ends of range must specify a breakpoint location, or neither can specify a breakpoint location.")

        self.expect("breakpoint disable 2.1 - 2.2",
                    startstr="2 breakpoints disabled.")

        self.expect("breakpoint enable 2.*",
                    patterns=[".* breakpoints enabled."])
