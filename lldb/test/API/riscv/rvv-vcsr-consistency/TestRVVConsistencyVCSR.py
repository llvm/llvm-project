"""
Test RISC-V RVV CSR (vtype, vcsr) consistency
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import rvvutil
from lldbsuite.test import lldbutil


class RISCVRVVConsistencyVCSRTestCase(TestBase):
    def _is_lmul_sew_legal(self):
        """Check if LMUL/SEW configuration is legal."""
        vlenb = rvvutil.get_vlenb(self)
        return vlenb * 8 * rvvutil.get_lmul(self) >= rvvutil.get_sew(self)

    def _get_required_vl(self):
        """Get vl value required by the application."""
        var = self.frame().FindVariable("vl")
        if not var.IsValid():
            return None
        error = lldb.SBError()
        return var.GetValueAsUnsigned(error)

    def _get_expected_vl(self):
        """Get vl value that expected to be set."""
        if not self._is_lmul_sew_legal():
            return 0
        return min(self._get_required_vl(), rvvutil.calculate_vlmax(self))

    def _get_vxml(self):
        vcsr = rvvutil.get_register_value(self, "vcsr")
        return (vcsr >> 1) & 0b11

    def _get_vxsat(self):
        vcsr = rvvutil.get_register_value(self, "vcsr")
        return vcsr & 0b1

    @skipIf(archs=no_match("^riscv.*"))
    def test_rvv_vl_register(self):
        """Test vl register for various configurations."""
        rvvutil.skip_if_rvv_unsupported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv"})

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "vsetvl_done", lldb.SBFileSpec("main.cpp")
        )

        finish_bp = lldbutil.run_break_set_by_source_regexp(self, "do_vsetv_test_end")
        finish_thread = lldbutil.get_one_thread_stopped_at_breakpoint_id(process, finish_bp)
        while finish_thread is None:
            if not self._is_lmul_sew_legal():
                # For illegal configs, vtype should show vill=1
                vtype = rvvutil.get_register_value(self, "vtype")
                self.assertEqual((vtype >> 63) & 1, 1, "vtype should show vill=1")

            self.expect("register read vl", substrs=[f"vl = 0x{self._get_expected_vl():0>16x}"])

            self.runCmd("continue")
            finish_thread = lldbutil.get_one_thread_stopped_at_breakpoint_id(process, finish_bp)

    @skipIf(archs=no_match("^riscv.*"))
    def test_rvv_vcsr_register(self):
        """Test vcsr register."""

        rvvutil.skip_if_rvv_unsupported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv"})
        main_source_file = lldb.SBFileSpec("main.cpp")

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self, "rvv_initialized", main_source_file)

        # Test different vxrm values
        for vxrm in range(4):
            lldbutil.continue_to_source_breakpoint(self, process, f"vxrm_{vxrm}", main_source_file)
            self.assertEqual(self._get_vxml(), vxrm, "Invalid vxrm value")

        lldbutil.continue_to_source_breakpoint(self, process, "vxrm_0_again", main_source_file)
        self.assertEqual(self._get_vxsat(), 1, "Invalid vxsat value")
        self.assertEqual(self._get_vxml(), 3, "Invalid vxrm value")

        lldbutil.continue_to_source_breakpoint(self, process, "vcsr_done", main_source_file)
        self.assertEqual(self._get_vxsat(), 1, "Invalid vxsat value")
        self.assertEqual(self._get_vxml(), 0, "Invalid vxrm value")
