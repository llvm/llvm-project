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

BranchOp mlir::edsc::intrinsics::std_br(Block *block, ValueRange operands) {
  return OperationBuilder<BranchOp>(block, operands);
}

CondBranchOp mlir::edsc::intrinsics::std_cond_br(Value cond, Block *trueBranch,
                                                 ValueRange trueOperands,
                                                 Block *falseBranch,
                                                 ValueRange falseOperands) {
  return OperationBuilder<CondBranchOp>(cond, trueBranch, trueOperands,
                                        falseBranch, falseOperands);
}
