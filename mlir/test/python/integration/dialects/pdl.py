# RUN: %PYTHON %s 2>&1 | FileCheck %s

from mlir.dialects import arith, func, pdl
from mlir.dialects.builtin import module
from mlir.ir import *
from mlir.rewrite import *


def construct_and_print_in_module(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            module = f(module)
        if module is not None:
            print(module)
    return f


# CHECK-LABEL: TEST: test_add_to_mul
# CHECK: arith.muli
@construct_and_print_in_module
def test_add_to_mul(module_):
    index_type = IndexType.get()

    # Create a test case.
    @module(sym_name="ir")
    def ir():
        @func.func(index_type, index_type)
        def add_func(a, b):
            return arith.addi(a, b)

    # Create a rewrite from add to mul. This will match
    # - operation name is arith.addi
    # - operands are index types.
    # - there are two operands.
    with Location.unknown():
        m = Module.create()
        with InsertionPoint(m.body):
            # Change all arith.addi with index types to arith.muli.
            @pdl.pattern(benefit=1, sym_name="addi_to_mul")
            def pat():
                # Match arith.addi with index types.
                index_type = pdl.TypeOp(IndexType.get())
                operand0 = pdl.OperandOp(index_type)
                operand1 = pdl.OperandOp(index_type)
                op0 = pdl.OperationOp(
                    name="arith.addi", args=[operand0, operand1], types=[index_type]
                )

                # Replace the matched op with arith.muli.
                @pdl.rewrite()
                def rew():
                    newOp = pdl.OperationOp(
                        name="arith.muli", args=[operand0, operand1], types=[index_type]
                    )
                    pdl.ReplaceOp(op0, with_op=newOp)

    # Create a PDL module from module and freeze it. At this point the ownership
    # of the module is transferred to the PDL module. This ownership transfer is
    # not yet captured Python side/has sharp edges. So best to construct the
    # module and PDL module in same scope.
    # FIXME: This should be made more robust.
    frozen = PDLModule(m).freeze()
    # Could apply frozen pattern set multiple times.
    apply_patterns_and_fold_greedily(module_, frozen)
    return module_
