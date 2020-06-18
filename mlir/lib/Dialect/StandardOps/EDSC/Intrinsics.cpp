//===- Intrinsics.cpp - MLIR Operations for Declarative Builders ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"

using namespace mlir;
using namespace mlir::edsc;

BranchOp mlir::edsc::intrinsics::std_br(BlockHandle bh, ValueRange operands) {
  assert(bh && "Expected already captured BlockHandle");
  SmallVector<Value, 4> ops(operands.begin(), operands.end());
  return OperationBuilder<BranchOp>(bh.getBlock(), ops);
}

BranchOp mlir::edsc::intrinsics::std_br(Block *block, ValueRange operands) {
  return OperationBuilder<BranchOp>(block, operands);
}

BranchOp mlir::edsc::intrinsics::std_br(BlockHandle *bh, ArrayRef<Type> types,
                                        MutableArrayRef<Value> captures,
                                        ValueRange operands) {
  assert(!*bh && "Unexpected already captured BlockHandle");
  BlockBuilder(bh, types, captures)(/* no body */);
  SmallVector<Value, 4> ops(operands.begin(), operands.end());
  return OperationBuilder<BranchOp>(bh->getBlock(), ops);
}

CondBranchOp mlir::edsc::intrinsics::std_cond_br(Value cond, Block *trueBranch,
                                                 ValueRange trueOperands,
                                                 Block *falseBranch,
                                                 ValueRange falseOperands) {
  return OperationBuilder<CondBranchOp>(cond, trueBranch, trueOperands,
                                        falseBranch, falseOperands);
}

CondBranchOp mlir::edsc::intrinsics::std_cond_br(Value cond,
                                                 BlockHandle trueBranch,
                                                 ValueRange trueOperands,
                                                 BlockHandle falseBranch,
                                                 ValueRange falseOperands) {
  SmallVector<Value, 4> trueOps(trueOperands.begin(), trueOperands.end());
  SmallVector<Value, 4> falseOps(falseOperands.begin(), falseOperands.end());
  return OperationBuilder<CondBranchOp>(cond, trueBranch.getBlock(), trueOps,
                                        falseBranch.getBlock(), falseOps);
}

CondBranchOp mlir::edsc::intrinsics::std_cond_br(
    Value cond, BlockHandle *trueBranch, ArrayRef<Type> trueTypes,
    MutableArrayRef<Value> trueCaptures, ValueRange trueOperands,
    BlockHandle *falseBranch, ArrayRef<Type> falseTypes,
    MutableArrayRef<Value> falseCaptures, ValueRange falseOperands) {
  assert(!*trueBranch && "Unexpected already captured BlockHandle");
  assert(!*falseBranch && "Unexpected already captured BlockHandle");
  BlockBuilder(trueBranch, trueTypes, trueCaptures)(/* no body */);
  BlockBuilder(falseBranch, falseTypes, falseCaptures)(/* no body */);
  SmallVector<Value, 4> trueOps(trueOperands.begin(), trueOperands.end());
  SmallVector<Value, 4> falseOps(falseOperands.begin(), falseOperands.end());
  return OperationBuilder<CondBranchOp>(cond, trueBranch->getBlock(), trueOps,
                                        falseBranch->getBlock(), falseOps);
}
