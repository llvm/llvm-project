"""
Test Getting the expression path for registers works correctly
"""

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbtest import TestBase, VALID_BREAKPOINT, VALID_TARGET


class TestExprPathRegisters(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def verify_register_path(self, reg_value: lldb.SBValue):
        stream = lldb.SBStream()
        reg_name = reg_value.name
        self.assertTrue(
            reg_value.GetExpressionPath(stream),
            f"Expected an expression path for register {reg_name}.",
        )
        reg_expr_path = stream.GetData()
        self.assertEqual(reg_expr_path, f"${reg_name}")

    def test_float_registers(self):
        """Verify the expression path of the registers is valid."""
        self.build()
        _, _, thread, _ = lldbutil.run_to_name_breakpoint(self, "my_foo")
        frame = thread.GetSelectedFrame()
        self.assertTrue(frame, "Expected a valid Frame.")

        # possible floating point register on some cpus.
        register_names = [
            "xmm0",
            "ymm0",
            "v0",
            "v1",
            "f0",
            "f1",
            "d0",
            "d1",
            "vr0",
            "vr1",
            "st0",
            "st1",
        ]
        for name in register_names:
            reg_value = frame.FindRegister(name)
            # some the register will not be available for the cpu
            # only verify if it is valid.
            if reg_value:
                self.verify_register_path(reg_value)

    def test_all_registers(self):
        """Test all the registers that is avaiable on the machine"""
        self.build()
        _, _, thread, _ = lldbutil.run_to_name_breakpoint(self, "my_foo")
        frame = thread.GetSelectedFrame()
        self.assertTrue(frame, "Expected a valid Frame.")

        register_sets = frame.GetRegisters()
        self.assertTrue(register_sets.IsValid(), "Expected Frame Registers")

        for register_set in register_sets:
            for register in register_set.children:
                self.verify_register_path(register)
