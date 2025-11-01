"""
Test RISC-V RVV register printing and formatting.
Tests basic 'register read' commands and register listing.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import rvvutil
from lldbsuite.test import lldbutil


class RISCVRVVPrintoutTestCase(TestBase):
    @skipIf(archs=no_match("^riscv.*"))
    def test_rvv_register_read(self):
        """Test basic vector register printing."""
        rvvutil.skip_if_rvv_unsupported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv"})

        lldbutil.run_to_source_breakpoint(self, "pre_vect_mem", lldb.SBFileSpec("main.cpp"))

        vlenb = rvvutil.get_vlenb(self)

        for reg_name in ["vstart", "vtype", "vcsr"]:
            self.expect(f"register read {reg_name}", substrs=[f"{reg_name} = 0x{0:0>16x}"])

        for reg_name in ["vl", "vlenb"]:
            self.expect(f"register read {reg_name}", substrs=[f"{reg_name} = 0x{vlenb:0>16x}"])

        rvvutil.check_vector_register_bytes(self, "v0", [0 for _ in range(vlenb)])
        rvvutil.check_vector_register_bytes(self, "v1", [1 for _ in range(vlenb)])
        rvvutil.check_vector_register_bytes(self, "v2", [3 for _ in range(vlenb)])

        for reg_num in range(3, 32):
            rvvutil.check_vector_register_bytes(self, f"v{reg_num}", [0 for _ in range(vlenb)])

    @skipIf(archs=no_match("^riscv.*"))
    def test_rvv_register_list(self):
        """Test 'register read --all' includes vector registers."""
        rvvutil.skip_if_rvv_unsupported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv"})

        lldbutil.run_to_source_breakpoint(self, "pre_vect_mem", lldb.SBFileSpec("main.cpp"))

        expected_regs = [f"v{i}" for i in range(32)]
        expected_regs += ["vtype", "vcsr", "vl", "vstart", "vlenb"]

        self.expect("register read --all")
        output = self.res.GetOutput()

        # Check no duplicates by counting occurrences
        for reg in expected_regs:
            count = output.count(f"{reg} ")
            self.assertEqual(count, 1)
