"""
Test RISC-V RVV register read/write operations.
Tests writing to vector registers and CSRs and verifying persistence.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import rvvutil
from lldbsuite.test import lldbutil


class RISCVRVVReadWriteTestCase(TestBase):
    def _run_check(self, reg_name, write_value, expected_value):
        self.runCmd(f"register write {reg_name} '{write_value}'")
        self.expect(f"register read {reg_name}", substrs=[f"{reg_name} = {expected_value}"])

        self.runCmd("stepi")
        self.expect(f"register read {reg_name}", substrs=[f"{reg_name} = {expected_value}"])

    @skipIf(archs=no_match("^riscv.*"))
    def test_rvv_write_vector_registers(self):
        """Test writing to vector registers v0-v31."""
        rvvutil.skip_if_rvv_unsupported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv"})

        lldbutil.run_to_source_breakpoint(self, "do_vector_stuff_end", lldb.SBFileSpec("main.cpp"))

        vlenb = rvvutil.get_vlenb(self)
        self.assertIsNotNone(vlenb, "vlenb should be readable")

        for reg_num in range(32):
            self.runCmd("continue")
            byte_str = f"{{{' '.join([f'0x{b:02x}' for b in list(range(vlenb))])}}}"
            self._run_check(f"v{reg_num}", byte_str, byte_str)

    @skipIf(archs=no_match("^riscv.*"))
    def test_rvv_write_vcsrs(self):
        """Test writing to CSRs."""
        rvvutil.skip_if_rvv_unsupported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv"})

        lldbutil.run_to_source_breakpoint(self, "do_vector_stuff_end", lldb.SBFileSpec("main.cpp"))

        for reg_name in ["vstart", "vtype", "vl"]:
            self.runCmd("continue")
            # 1 should be a valid value for vstart, vtype, vl
            # so we use it here
            self._run_check(reg_name, 1, f"0x{1:0>16x}")

        # vlenb is read only
        self.runCmd("continue")
        self.expect("register write vlenb 1", substrs=["Failed to write register"], error=True)
