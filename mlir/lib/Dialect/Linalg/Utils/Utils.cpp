//===- Utils.cpp - Utilities to support the Linalg dialect ----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Utils/Intrinsics.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/FoldUtils.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::linalg::intrinsics;
using namespace mlir::loop;

static Value emitOrFoldComposedAffineApply(OpBuilder &b, Location loc,
                                           AffineMap map,
                                           ArrayRef<Value> operandsRef,
                                           OperationFolder *folder) {
  SmallVector<Value, 4> operands(operandsRef.begin(), operandsRef.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  canonicalizeMapAndOperands(&map, &operands);
  return folder ? folder->create<AffineApplyOp>(b, loc, map, operands)
                : b.create<AffineApplyOp>(loc, map, operands);
}

SmallVector<Value, 4> mlir::linalg::applyMapToValues(OpBuilder &b, Location loc,
                                                     AffineMap map,
                                                     ArrayRef<Value> values,
                                                     OperationFolder *folder) {
  SmallVector<Value, 4> res;
  res.reserve(map.getNumResults());
  unsigned numDims = map.getNumDims();
  // For each `expr` in `map`, applies the `expr` to the values extracted from
  // ranges. If the resulting application can be folded into a Value, the
  // folding occurs eagerly. Otherwise, an affine.apply operation is emitted.
  for (auto expr : map.getResults()) {
    AffineMap map = AffineMap::get(numDims, 0, expr);
    res.push_back(emitOrFoldComposedAffineApply(b, loc, map, values, folder));
  }
  return res;
}

/// Returns all the operands of `linalgOp` that are not views.
/// Asserts that these operands are value types to allow transformations like
/// tiling to just use the values when cloning `linalgOp`.
SmallVector<Value, 4>
mlir::linalg::getAssumedNonViewOperands(LinalgOp linalgOp) {
  auto *op = linalgOp.getOperation();
  unsigned numViews = linalgOp.getNumInputsAndOutputs();
  unsigned nOperands = op->getNumOperands() - numViews;
  SmallVector<Value, 4> res;
  res.reserve(nOperands);
  for (unsigned i = 0; i < nOperands; ++i) {
    res.push_back(op->getOperand(numViews + i));
    auto t = res.back().getType();
    (void)t;
    assert((t.isIntOrIndexOrFloat() || t.isa<VectorType>()) &&
           "expected scalar or vector type");
  }
  return res;
}
