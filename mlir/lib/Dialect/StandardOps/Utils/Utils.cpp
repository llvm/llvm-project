//===- Utils.cpp - Utilities to support the Linalg dialect ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/Utils/Utils.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;

/// Matches a ConstantIndexOp.
/// TODO: This should probably just be a general matcher that uses matchConstant
/// and checks the operation for an index type.
detail::op_matcher<ConstantIndexOp> mlir::matchConstantIndex() {
  return detail::op_matcher<ConstantIndexOp>();
}

/// Detects the `values` produced by a ConstantIndexOp and places the new
/// constant in place of the corresponding sentinel value.
void mlir::canonicalizeSubViewPart(
    SmallVectorImpl<OpFoldResult> &values,
    llvm::function_ref<bool(int64_t)> isDynamic) {
  for (OpFoldResult &ofr : values) {
    if (ofr.is<Attribute>())
      continue;
    // Newly static, move from Value to constant.
    if (auto cstOp = ofr.dyn_cast<Value>().getDefiningOp<ConstantIndexOp>())
      ofr = OpBuilder(cstOp).getIndexAttr(cstOp.getValue());
  }
}

void mlir::getPositionsOfShapeOne(
    unsigned rank, ArrayRef<int64_t> shape,
    llvm::SmallDenseSet<unsigned> &dimsToProject) {
  dimsToProject.reserve(rank);
  for (unsigned pos = 0, e = shape.size(); pos < e && rank > 0; ++pos) {
    if (shape[pos] == 1) {
      dimsToProject.insert(pos);
      --rank;
    }
  }
}

Value ArithBuilder::_and(Value lhs, Value rhs) {
  return b.create<AndOp>(loc, lhs, rhs);
}
Value ArithBuilder::add(Value lhs, Value rhs) {
  if (lhs.getType().isa<IntegerType>())
    return b.create<AddIOp>(loc, lhs, rhs);
  return b.create<AddFOp>(loc, lhs, rhs);
}
Value ArithBuilder::mul(Value lhs, Value rhs) {
  if (lhs.getType().isa<IntegerType>())
    return b.create<MulIOp>(loc, lhs, rhs);
  return b.create<MulFOp>(loc, lhs, rhs);
}
Value ArithBuilder::sgt(Value lhs, Value rhs) {
  if (lhs.getType().isa<IndexType, IntegerType>())
    return b.create<CmpIOp>(loc, CmpIPredicate::sgt, lhs, rhs);
  return b.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
}
Value ArithBuilder::slt(Value lhs, Value rhs) {
  if (lhs.getType().isa<IndexType, IntegerType>())
    return b.create<CmpIOp>(loc, CmpIPredicate::slt, lhs, rhs);
  return b.create<CmpFOp>(loc, CmpFPredicate::OLT, lhs, rhs);
}
Value ArithBuilder::select(Value cmp, Value lhs, Value rhs) {
  return b.create<SelectOp>(loc, cmp, lhs, rhs);
}
