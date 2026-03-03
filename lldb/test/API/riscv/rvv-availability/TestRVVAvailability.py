"""
Test RISC-V RVV register availability.
Tests that vector registers are unavailable before any vector instruction
is executed.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import rvvutil
from lldbsuite.test import lldbutil


class RISCVRVVAvailabilityTestCase(TestBase):
    def _check_vector_registers_unavailable(self):
        """Check that all vector registers and CSRs are unavailable."""
        for reg_name in ["vstart", "vl", "vtype", "vcsr", "vlenb", "v0", "v1", "v15", "v31"]:
            self.expect(f"register read {reg_name}", substrs=["error:", "unavailable"])

    def _check_vector_registers_available(self):
        """Check that all vector registers and CSRs are available."""
        for reg_name in ["vstart", "vl", "vtype", "vcsr", "vlenb", "v0", "v1", "v15", "v31"]:
            self.expect(f"register read {reg_name}", substrs=[f"{reg_name} = "])

    @skipIf(archs=no_match("^riscv.*"))
    def test_available_with_vlenb_read(self):
        """Test registers available even when only vlenb is read before main."""
        rvvutil.skip_if_rvv_unsupported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv -DREAD_VLENB_BEFORE_MAIN"})
        lldbutil.run_to_name_breakpoint(self, "main")
        self._check_vector_registers_available()

    @skipIf(archs=no_match("^riscv.*"))
    def test_unavailable_without_vector_ops(self):
        """Test registers unavailable when no vector operations performed."""
        rvvutil.skip_if_rvv_unsupported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv"})
        lldbutil.run_to_name_breakpoint(self, "main")
        self._check_vector_registers_unavailable()

    @skipIf(archs=no_match("^riscv.*"))
    def test_available_after_vsetvli(self):
        """Test registers available after vsetvli is executed."""
        rvvutil.skip_if_rvv_unsupported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv -DSET_VSETVLI_BEFORE_MAIN"})
        lldbutil.run_to_name_breakpoint(self, "main")
        self._check_vector_registers_available()
