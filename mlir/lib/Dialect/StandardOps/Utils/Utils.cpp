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
