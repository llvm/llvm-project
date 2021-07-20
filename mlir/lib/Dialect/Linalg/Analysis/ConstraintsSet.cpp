//===- ConstraintsSet.cpp - Extensions for FlatAffineConstraints ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Linalg-specific constraints set extensions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Analysis/ConstraintsSet.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;

unsigned ConstraintsSet::lookupPos(Value id) const {
  unsigned pos;
  if (!findId(id, &pos)) {
    llvm::errs() << "Lookup failed: " << id << "\n";
    llvm_unreachable("Lookup failed");
  }
  return pos;
}

LogicalResult ConstraintsSet::ensureIdOfType(Value v, bool asDim) {
  if (!containsId(v)) {
    if (asDim)
      addDimId(getNumDimIds(), v);
    else
      addSymbolId(getNumSymbolIds(), v);
    return success();
  }
  unsigned pos = lookupPos(v);
  return success((asDim && pos < getNumDimIds()) ||
                 (!asDim && getNumDimIds() <= pos &&
                  pos < getNumDimIds() + getNumSymbolIds()));
}

LogicalResult ConstraintsSet::composeAffineApply(Value val, AffineMap map,
                                                 ValueRange operands) {
  AffineValueMap avm(map, operands, val);
  return composeMap(&avm);
}

LogicalResult ConstraintsSet::composeMinOrMaxMapAndOperands(Value val,
                                                            AffineMap map,
                                                            ValueRange operands,
                                                            bool min) {
  ConstraintsSet localCst;
  std::vector<SmallVector<int64_t, 8>> flatExprs;
  if (failed(getFlattenedAffineExprs(map, &flatExprs, &localCst)))
    return failure();
  assert(flatExprs.size() == map.getNumResults() &&
         "incorrect number of flattened expressiosn");

  // Local vars on a per-need basis.
  if (localCst.getNumLocalIds() != 0)
    return failure();

  // Add one inequality for each result connecting `val` to the other ids in
  // `operands`. For instance, uf the expression is:
  //   `16 * i0 + i1` and
  //   `min` is true
  // add:
  //  -d_val + 16 * i0 + i1 >= 0.
  for (const auto &flatExpr : flatExprs) {
    assert(flatExpr.size() >= operands.size() + 1);
    SmallVector<int64_t, 8> ineq(getNumCols(), 0);
    for (unsigned i = 0, e = operands.size(); i < e; i++)
      ineq[lookupPos(operands[i])] = min ? flatExpr[i] : -flatExpr[i];

    // Set the coefficient for `d_val`.
    ineq[lookupPos(val)] = min ? -1 : 1;

    // Set the constant term (upper bound in flatExpr is exclusive).
    ineq[getNumCols() - 1] = min ? flatExpr[flatExpr.size() - 1] - 1
                                 : -flatExpr[flatExpr.size() - 1];

    // Add the inequality connecting the result of the map to the rest.
    addInequality(ineq);
  }

  return success();
}
