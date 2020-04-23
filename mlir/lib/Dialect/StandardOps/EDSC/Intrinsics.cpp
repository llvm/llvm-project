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

OperationHandle mlir::edsc::intrinsics::std_br(BlockHandle bh,
                                               ArrayRef<Value> operands) {
  assert(bh && "Expected already captured BlockHandle");
  for (auto &o : operands) {
    (void)o;
    assert(o && "Expected already captured Value");
  }
  SmallVector<Value, 4> ops(operands.begin(), operands.end());
  return OperationHandle::create<BranchOp>(bh.getBlock(), ops);
}

OperationHandle mlir::edsc::intrinsics::std_br(BlockHandle *bh,
                                               ArrayRef<Type> types,
                                               MutableArrayRef<Value> captures,
                                               ArrayRef<Value> operands) {
  assert(!*bh && "Unexpected already captured BlockHandle");
  BlockBuilder(bh, types, captures)(/* no body */);
  SmallVector<Value, 4> ops(operands.begin(), operands.end());
  return OperationHandle::create<BranchOp>(bh->getBlock(), ops);
}

OperationHandle mlir::edsc::intrinsics::std_cond_br(
    Value cond, BlockHandle trueBranch, ArrayRef<Value> trueOperands,
    BlockHandle falseBranch, ArrayRef<Value> falseOperands) {
  SmallVector<Value, 4> trueOps(trueOperands.begin(), trueOperands.end());
  SmallVector<Value, 4> falseOps(falseOperands.begin(), falseOperands.end());
  return OperationHandle::create<CondBranchOp>(
      cond, trueBranch.getBlock(), trueOps, falseBranch.getBlock(), falseOps);
}

OperationHandle mlir::edsc::intrinsics::std_cond_br(
    Value cond, BlockHandle *trueBranch, ArrayRef<Type> trueTypes,
    MutableArrayRef<Value> trueCaptures, ArrayRef<Value> trueOperands,
    BlockHandle *falseBranch, ArrayRef<Type> falseTypes,
    MutableArrayRef<Value> falseCaptures, ArrayRef<Value> falseOperands) {
  assert(!*trueBranch && "Unexpected already captured BlockHandle");
  assert(!*falseBranch && "Unexpected already captured BlockHandle");
  BlockBuilder(trueBranch, trueTypes, trueCaptures)(/* no body */);
  BlockBuilder(falseBranch, falseTypes, falseCaptures)(/* no body */);
  SmallVector<Value, 4> trueOps(trueOperands.begin(), trueOperands.end());
  SmallVector<Value, 4> falseOps(falseOperands.begin(), falseOperands.end());
  return OperationHandle::create<CondBranchOp>(
      cond, trueBranch->getBlock(), trueOps, falseBranch->getBlock(), falseOps);
}
