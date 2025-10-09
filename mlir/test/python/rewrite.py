# RUN: %PYTHON %s 2>&1 | FileCheck %s

import gc, sys
from mlir.ir import *
from mlir.passmanager import *
from mlir.dialects.builtin import ModuleOp
from mlir.dialects import arith
from mlir.rewrite import *


def log(*args):
    print(*args, file=sys.stderr)
    sys.stderr.flush()


def run(f):
    log("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0

# CHECK-LABEL: TEST: testRewritePattern
@run
def testRewritePattern():
    def to_muli(op, rewriter, pattern):
        with rewriter.ip:
            new_op = arith.muli(op.operands[0], op.operands[1], loc=op.location)
        rewriter.replace_op(op, new_op.owner)

    with Context():
        patterns = RewritePatternSet()
        patterns.add(arith.AddIOp, to_muli)
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
