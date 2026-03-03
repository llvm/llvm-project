"""
Test RISC-V RVV register modification side effects.
Tests that modifying vector state (vl, vtype) affects program execution correctly.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import rvvutil
from lldbsuite.test import lldbutil


class RISCVRVVSideEffectsTestCase(TestBase):
    @skipIf(archs=no_match("^riscv.*"))
    def test_rvv_controlled_vadd(self):
        """Test vector addition with modified register values."""

        rvvutil.skip_if_rvv_unsupported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv"})

        main_source_file = lldb.SBFileSpec("main.cpp")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "vect_control_vadd_start", main_source_file
        )

        vlenb = rvvutil.get_vlenb(self)
        self.assertIsNotNone(vlenb, "VLENB should be readable")

        for i in range(32):
            rvvutil.check_vector_register_bytes(self, f"v{i}", [0 for _ in range(vlenb)])

        rvvutil.set_vector_register_bytes(self, "v0", list(range(vlenb)))
        rvvutil.set_vector_register_bytes(self, "v1", [i + 7 for i in range(vlenb)])

        # Continue to after vadd
        lldbutil.continue_to_source_breakpoint(self, process, "controlled_vadd_done", main_source_file)

        # Check that v0 and v1 are unchanged
        rvvutil.check_vector_register_bytes(self, "v0", list(range(vlenb)))
        rvvutil.check_vector_register_bytes(self, "v1", [i + 7 for i in range(vlenb)])

        # Check that v2 contains the sum (v0 + v1)
        rvvutil.check_vector_register_bytes(self, "v2", [i + i + 7 for i in range(vlenb)])

        for i in range(3, 32):
            rvvutil.check_vector_register_bytes(self, f"v{i}", [0 for _ in range(vlenb)])

    @skipIf(archs=no_match("^riscv.*"))
    def test_rvv_wide_operations_with_vl_modification(self):
        """Test that modifying vl affects subsequent operations."""

        rvvutil.skip_if_rvv_unsupported(self)
        self.build(dictionary={"CFLAGS_EXTRAS": "-march=rv64gcv"})

        main_source_file = lldb.SBFileSpec("main.cpp")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self, "vect_op_v0_add1", main_source_file)

        self.runCmd("register write vl 2")

        lldbutil.continue_to_source_breakpoint(self, process, "vect_op_v24_v0_add2", main_source_file)

        vlenb = rvvutil.get_vlenb(self)
        self.assertIsNotNone(vlenb, "VLENB should be readable")

        rvvutil.set_vector_register_bytes(self, "v8", list(range(vlenb)))

        lldbutil.continue_to_source_breakpoint(self, process, "vect_op_v16_v8_add2", main_source_file)

        self.runCmd("register write vtype 0")

        lldbutil.continue_to_source_breakpoint(self, process, "vect_wide_op_end", main_source_file)

        self.expect("register read vtype", substrs=[f"vtype = 0x{0:0>16x}"])
        self.expect("register read vl", substrs=[f"vl = 0x{2:0>16x}"])

        rvvutil.check_vector_register_bytes(self, "v8", list(range(vlenb)))

        # Check that operations only affected first 2 elements (vl=2)
        # v10 and v24 should have [3, 3, 0, 0, ...]
        v10_value = [0 for _ in range(vlenb)]
        v10_value[0:2] = [3, 3]
        rvvutil.check_vector_register_bytes(self, "v10", v10_value)
        rvvutil.check_vector_register_bytes(self, "v24", v10_value)

        # v16 should have [2, 3, 0, 0, ...]
        v16_value = [0 for _ in range(vlenb)]
        v16_value[0:2] = [2, 3]
        rvvutil.check_vector_register_bytes(self, "v16", v16_value)
