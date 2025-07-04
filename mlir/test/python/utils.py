# RUN: %python %s | FileCheck %s
# RUN: %python %s 2>&1 | FileCheck %s --check-prefix=DEBUG_ONLY

import unittest

from mlir import ir
from mlir.passmanager import PassManager
from mlir.dialects import arith
from mlir.utils import (
    call_with_toplevel_context_create_module,
    caller_mlir_context,
    debug_conversion,
    using_mlir_context,
)


class TestRequiredContext(unittest.TestCase):
    def test_shared_context(self):
        """Test that the context is reused, so values can be passed/returned between functions."""

        @using_mlir_context()
        def create_add(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
            return arith.AddFOp(
                lhs, rhs, fastmath=arith.FastMathFlags.nnan | arith.FastMathFlags.ninf
            ).result

        @using_mlir_context()
        def multiple_adds(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
            return create_add(create_add(lhs, rhs), create_add(lhs, rhs))

        @call_with_toplevel_context_create_module
        def _(module) -> None:
            c = arith.ConstantOp(value=42.42, result=ir.F32Type.get()).result
            multiple_adds(c, c)

            # CHECK-LABEL: module {
            # CHECK: constant
            # CHECK-NEXT: arith.addf
            # CHECK-NEXT: arith.addf
            # CHECK-NEXT: arith.addf
            print(module)

    def test_unregistered_op_asserts(self):
        """Confirm that with_mlir_context fails if an operation is still not registered."""
        with self.assertRaises(AssertionError), using_mlir_context(
            required_extension_operations=["func.fake_extension_op"],
            registration_funcs=[],
        ):
            pass

    def test_required_op_asserts(self):
        """Confirm that with_mlir_context fails if an operation is still not registered."""
        with self.assertRaises(AssertionError), caller_mlir_context(
            required_extension_operations=["func.fake_extension_op"],
            registration_funcs=[],
        ):
            pass


class TestDebugOnlyFlags(unittest.TestCase):
    def test_debug_types(self):
        """Test checks --debug-only=xxx functionality is available in MLIR."""

        @debug_conversion()
        def lower(module) -> None:
            pm = PassManager("builtin.module")
            pm.add("convert-arith-to-llvm")
            pm.run(module.operation)

        @call_with_toplevel_context_create_module
        def _(module) -> None:
            c = arith.ConstantOp(value=42.42, result=ir.F32Type.get()).result
            arith.AddFOp(c, c, fastmath=arith.FastMathFlags.nnan | arith.FastMathFlags.ninf)

            # DEBUG_ONLY-LABEL: Legalizing operation : 'builtin.module'
            #       DEBUG_ONLY: Legalizing operation : 'arith.addf'
            lower(module)


if __name__ == "__main__":
    unittest.main()
