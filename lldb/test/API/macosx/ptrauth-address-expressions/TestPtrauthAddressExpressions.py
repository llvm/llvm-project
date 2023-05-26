"""Test that AArch64 PAC bits are stripped from address expression arguments"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPtrauthAddressExpressions(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # On Darwin systems, arch arm64e means ARMv8.3 with ptrauth
    # ABI used.
    @skipIf(archs=no_match(["arm64e"]))
    def test(self):
        # Skip this test if not running on AArch64 target that supports PAC
        if not self.isAArch64PAuth():
            self.skipTest("Target must support pointer authentication.")
        self.source = "main.c"
        self.build()
        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source, False)
        )

        self.expect("p fptr", substrs=[self.source])
        self.expect("ima loo -va fptr", substrs=[self.source])
        self.expect("break set -a fptr", substrs=[self.source])
