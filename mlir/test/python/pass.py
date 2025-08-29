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

        module = ModuleOp.parse(r"""
            module {
              func.func @add(%a: index, %b: index) -> index {
                %sum = arith.addi %a, %b : index
                return %sum : index
              }
            }
        """)

        # CHECK-LABEL: Dump After CustomPass
        # CHECK: arith.muli
        pm = PassManager('any')
        pm.enable_ir_printing()
        pm.add(CustomPass())
        pm.run(module)
