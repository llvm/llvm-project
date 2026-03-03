"""
Test RISC-V RVV behavior on targets without RVV support.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import rvvutil
from lldbsuite.test import lldbutil


class RISCVRVVUnsupportedTestCase(TestBase):
    # This test should only run on RISC-V targets without RVV support
    @skipIf(archs=no_match("^riscv.*"))
    def test_rvv_unsupported(self):
        """Test that vector registers are inaccessible on non-RVV targets."""
        rvvutil.skip_if_rvv_supported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv"})

        lldbutil.run_to_source_breakpoint(self, "break", lldb.SBFileSpec("main.cpp"))

        self.expect("print a", substrs=["42"])
        self.runCmd("register write a0 42")
        self.expect("register read a0", substrs=[f"0x{42:0>16x}"])

        for reg_name in ["vstart", "vl", "vlenb", "v0", "v15", "v31"]:
            self.expect(f"register read {reg_name}", substrs=["error:", "Invalid register name"], error=True)

        # Basic debugging should still work
        self.expect("print a", substrs=["42"])
        self.runCmd("register write a0 43")
        self.expect("register read a0", substrs=[f"0x{43:0>16x}"])
