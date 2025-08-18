//===- LowerVectorFromElements.cpp - Lower 'vector.from_elements' op -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.from_elements' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"

#define DEBUG_TYPE "lower-vector-from-elements"

using namespace mlir;

namespace {

/// Unrolls 2 or more dimensional `vector.from_elements` ops by unrolling the
/// outermost dimension. For example:
/// ```
/// %v = vector.from_elements %e0, %e1, %e2, %e3, %e4, %e5 : vector<2x3xf32>
///
/// ==>
///
/// %0   = ub.poison : vector<2x3xf32>
/// %v0  = vector.from_elements %e0, %e1, %e2 : vector<3xf32>
/// %1   = vector.insert %v0, %0 [0] : vector<3xf32> into vector<2x3xf32>
/// %v1  = vector.from_elements %e3, %e4, %e5 : vector<3xf32>
/// %v   = vector.insert %v1, %1 [1] : vector<3xf32> into vector<2x3xf32>
/// ```
///
/// When applied exhaustively, this will produce a sequence of 1-d from_elements
/// ops.
struct UnrollFromElements : OpRewritePattern<vector::FromElementsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::FromElementsOp op,
                                PatternRewriter &rewriter) const override {
    ValueRange allElements = op.getElements();

    auto unrollFromElementsFn = [&](PatternRewriter &rewriter, Location loc,
                                    VectorType subTy, int64_t index) {
      size_t subTyNumElements = subTy.getNumElements();
      assert((index + 1) * subTyNumElements <= allElements.size() &&
             "out of bounds");
      ValueRange subElements =
          allElements.slice(index * subTyNumElements, subTyNumElements);
      return vector::FromElementsOp::create(rewriter, loc, subTy, subElements);
    };

    return unrollVectorOp(op, rewriter, unrollFromElementsFn);
  }
};

} // namespace

void mlir::vector::populateVectorFromElementsLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<UnrollFromElements>(patterns.getContext(), benefit);
}
