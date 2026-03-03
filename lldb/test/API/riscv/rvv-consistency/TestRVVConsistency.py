"""
Test RISC-V RVV state consistency and edge cases.
Tests overflow handling, illegal configurations, and state coherence.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import rvvutil
from lldbsuite.test import lldbutil


class RISCVRVVConsistencyTestCase(TestBase):
    @skipIf(archs=no_match("^riscv.*"))
    def test_rvv_vl_overflow(self):
        """Test that vl is clamped to VLMAX when set too high."""

        rvvutil.skip_if_rvv_unsupported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv"})

        lldbutil.run_to_source_breakpoint(self, "workload_end", lldb.SBFileSpec("main.cpp"))

        # Try to set vl to a huge value (9999)
        self.runCmd("register write vl 9999")
        self.expect("register read vl", substrs=[f"vl = 0x{9999:0>16x}"])

        # Continue and check that vl is clamped
        self.runCmd("continue")

        self.expect("register read vl", substrs=[f"vl = 0x{rvvutil.calculate_vlmax(self):0>16x}"])

    @skipIf(archs=no_match("^riscv.*"))
    def test_rvv_vl_lmul_coherence(self):
        """Test that changing LMUL affects vl appropriately."""

        rvvutil.skip_if_rvv_unsupported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv"})

        lldbutil.run_to_source_breakpoint(self, "workload_end", lldb.SBFileSpec("main.cpp"))

        self.runCmd("register write vtype 3")
        self.expect("register read vtype", substrs=[f"vtype = 0x{3:0>16x}"])
        self.runCmd("register write vl 9999")
        self.expect("register read vl", substrs=[f"vl = 0x{9999:0>16x}"])

        self.runCmd("continue")

        self.assertEqual(rvvutil.get_lmul(self), 8, "LMUL should be 8")

        vlenb = rvvutil.get_vlenb(self)
        self.expect("register read vl", substrs=[f"vl = 0x{rvvutil.calculate_vlmax(self):0>16x}"])

        self.runCmd("continue")

        # Restore LMUL=1
        self.runCmd("register write vtype 0")
        self.expect("register read vtype", substrs=[f"vtype = 0x{0:0>16x}"])

        self.runCmd("continue")

        self.assertEqual(rvvutil.get_lmul(self), 1, "LMUL should be 1")

        # Check that vl is clamped after the LMUL was reduced
        self.expect("register read vl", substrs=[f"vl = 0x{rvvutil.calculate_vlmax(self):0>16x}"])

    @skipIf(archs=no_match("^riscv.*"))
    def test_rvv_vstart(self):
        """Test vstart behavior."""

        rvvutil.skip_if_rvv_unsupported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv"})

        lldbutil.run_to_source_breakpoint(self, "workload_end", lldb.SBFileSpec("main.cpp"))

        self.runCmd("register write vstart 8")
        self.expect("register read vstart", substrs=[f"vstart = 0x{8:0>16x}"])

        self.runCmd("continue")

        # vstart should be cleared to 0 by vector instructions
        self.expect("register read vstart", substrs=[f"vstart = 0x{0:0>16x}"])

    @skipIf(archs=no_match("^riscv.*"))
    def test_rvv_illegal_vtype(self):
        """Test illegal vtype configuration handling."""
        rvvutil.skip_if_rvv_unsupported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv"})

        lldbutil.run_to_source_breakpoint(self, "workload_end", lldb.SBFileSpec("main.cpp"))

        vlenb = rvvutil.get_vlenb(self)
        if vlenb >= 64:
            self.skipTest("Test requires VLENB < 64")

        self.runCmd("continue")

        # Set illegal vtype: SEW=64 (vsew=3), LMUL=1/8 (vlmul=5)
        # vtype = 5 | (3 << 3) = 5 | 24 = 29
        illegal_vtype = 5 | (3 << 3)
        self.runCmd(f"register write vtype {illegal_vtype}")
        self.assertEqual(rvvutil.get_sew(self), 64, "SEW should be 64")
        self.assertAlmostEqual(rvvutil.get_lmul(self), 1 / 8, places=7, msg="LMUL should be 1/8")

        self.runCmd("stepi")

        vtype = rvvutil.get_register_value(self, "vtype")
        self.assertEqual((vtype >> 63) & 1, 1, "vtype should have vill=1")

        # Legalize vtype
        self.runCmd("register write vtype 0")
        self.expect("register read vtype", substrs=[f"vtype = 0x{0:0>16x}"])

        self.runCmd("continue")

        self.expect("register read vtype", substrs=[f"vtype = 0x{0:0>16x}"])
