# RUN: %PYTHON %s 2>&1 | FileCheck %s

from mlir.ir import *
from mlir.passmanager import *
from mlir.dialects.builtin import ModuleOp
from mlir.dialects import arith
from mlir.rewrite import *


def run(f):
    print("\nTEST:", f.__name__)
    f()


# CHECK-LABEL: TEST: testRewritePattern
@run
def testRewritePattern():
    def to_muli(op, rewriter):
        with rewriter.ip:
            new_op = arith.muli(op.operands[0], op.operands[1], loc=op.location)
        rewriter.replace_op(op, new_op.owner)

    def constant_1_to_2(op, rewriter):
        c = op.attributes["value"].value
        if c != 1:
            return True  # failed to match
        with rewriter.ip:
            new_op = arith.constant(op.result.type, 2, loc=op.location)
        rewriter.replace_op(op, [new_op])

    with Context():
        patterns = RewritePatternSet()
        patterns.add(arith.AddIOp, to_muli)
        patterns.add(arith.ConstantOp, constant_1_to_2)
        frozen = patterns.freeze()

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

        apply_patterns_and_fold_greedily(module, frozen)
        # CHECK: %0 = arith.muli %arg0, %arg1 : i64
        # CHECK: return %0 : i64
        print(module)

        module = ModuleOp.parse(
            r"""
            module {
              func.func @const() -> (i64, i64) {
                %0 = arith.constant 1 : i64
                %1 = arith.constant 3 : i64
                return %0, %1 : i64, i64
              }
            }
            """
        )

        apply_patterns_and_fold_greedily(module, frozen)
        # CHECK: %c2_i64 = arith.constant 2 : i64
        # CHECK: %c3_i64 = arith.constant 3 : i64
        # CHECK: return %c2_i64, %c3_i64 : i64, i64
        print(module)
