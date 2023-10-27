# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import cf


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = Module.create()
        with InsertionPoint(module.body):
            f()
    return f


# CHECK-LABEL: TEST: testBranchAndSetSuccessor
@constructAndPrintInModule
def testBranchAndSetSuccessor():
    op1 = Operation.create("custom.op1", regions=1)

    block0 = op1.regions[0].blocks.append()
    ip = InsertionPoint(block0)
    Operation.create("custom.terminator", ip=ip)

    block1 = op1.regions[0].blocks.append()
    ip = InsertionPoint(block1)
    br1 = cf.BranchOp([], block1, ip=ip)
    # CHECK: ^bb1:  // pred: ^bb1
    # CHECK:   cf.br ^bb1
    print(br1.successors[0])
    # CHECK: num_successors 1
    print("num_successors", len(br1.successors))

    block2 = op1.regions[0].blocks.append()
    ip = InsertionPoint(block2)
    br2 = cf.BranchOp([], block1, ip=ip)
    # CHECK: ^bb1:  // 2 preds: ^bb1, ^bb2
    # CHECK:   cf.br ^bb1
    print(br2.successors[0])
    # CHECK: num_successors 1
    print("num_successors", len(br2.successors))

    br1.successors[0] = block2
    # CHECK: ^bb2:  // pred: ^bb1
    # CHECK:   cf.br ^bb1
    print(br1.successors[0])
    # CHECK: ^bb1:  // pred: ^bb2
    # CHECK:   cf.br ^bb2
    print(br2.operation.successors[0])
