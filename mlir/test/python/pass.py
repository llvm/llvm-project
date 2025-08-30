# RUN: %PYTHON %s 2>&1 | FileCheck %s

import gc, sys
from mlir.ir import *
from mlir.passmanager import *
from mlir.dialects.builtin import ModuleOp
from mlir.dialects import pdl
from mlir.rewrite import *


def log(*args):
    print(*args, file=sys.stderr)
    sys.stderr.flush()


def run(f):
    log("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0


def make_pdl_module():
    with Location.unknown():
        pdl_module = Module.create()
        with InsertionPoint(pdl_module.body):
            # Change all arith.addi with index types to arith.muli.
            @pdl.pattern(benefit=1, sym_name="addi_to_mul")
            def pat():
                # Match arith.addi with index types.
                i64_type = pdl.TypeOp(IntegerType.get_signless(64))
                operand0 = pdl.OperandOp(i64_type)
                operand1 = pdl.OperandOp(i64_type)
                op0 = pdl.OperationOp(
                    name="arith.addi", args=[operand0, operand1], types=[i64_type]
                )

                # Replace the matched op with arith.muli.
                @pdl.rewrite()
                def rew():
                    newOp = pdl.OperationOp(
                        name="arith.muli", args=[operand0, operand1], types=[i64_type]
                    )
                    pdl.ReplaceOp(op0, with_op=newOp)

        return pdl_module


# CHECK-LABEL: TEST: testCustomPass
@run
def testCustomPass():
    with Context():
        pdl_module = make_pdl_module()

        class CustomPass(Pass):
            def __init__(self):
                super().__init__("CustomPass", op_name="builtin.module")

            def run(self, m):
                frozen = PDLModule(pdl_module).freeze()
                apply_patterns_and_fold_greedily_for_op(m, frozen)

        module = ModuleOp.parse(
            r"""
            module {
              func.func @add(%a: i64, %b: i64) -> i64 {
                %sum = arith.addi %a, %b : i64
                return %sum : i64
              }
            }
        """
        )

        pm = PassManager("any")
        pm.enable_ir_printing()

        # CHECK-LABEL: Dump After CustomPass
        # CHECK: arith.muli
        pm.add(CustomPass())
        # CHECK-LABEL: Dump After ArithToLLVMConversionPass
        # CHECK: llvm.mul
        pm.add("convert-arith-to-llvm")
        pm.run(module)
