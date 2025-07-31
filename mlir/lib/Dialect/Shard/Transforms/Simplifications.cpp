//===- Simplifications.cpp - Shard Simplifications -_------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Shard/Transforms/Simplifications.h"
#include "TransformsDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <numeric>

namespace mlir {
namespace shard {

void populateSimplificationPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection) {
  populateAllReduceEndomorphismSimplificationPatterns<arith::AddFOp>(
      patterns, ReductionKind::Sum);
  populateAllReduceEndomorphismSimplificationPatterns<arith::AddIOp>(
      patterns, ReductionKind::Sum);

  populateAllReduceEndomorphismSimplificationPatterns<arith::MinimumFOp>(
      patterns, ReductionKind::Min);
  populateAllReduceEndomorphismSimplificationPatterns<arith::MinSIOp>(
      patterns, ReductionKind::Min);
  populateAllReduceEndomorphismSimplificationPatterns<arith::MinUIOp>(
      patterns, ReductionKind::Min);

  populateAllReduceEndomorphismSimplificationPatterns<arith::MaximumFOp>(
      patterns, ReductionKind::Max);
  populateAllReduceEndomorphismSimplificationPatterns<arith::MaxSIOp>(
      patterns, ReductionKind::Max);
  populateAllReduceEndomorphismSimplificationPatterns<arith::MaxUIOp>(
      patterns, ReductionKind::Max);

  // TODO: add simplifications for all-gather and other collectives.

  populateFoldingPatterns(patterns, symbolTableCollection);
}

namespace {

// This folding can not be done with an operation's fold method or
// DialectFoldInterface, because it needs a SymbolTableCollection to cache the
// symbol tables.
// We can't use DialectFoldInterface since the cache may be invalidated by some
// pass changing the referenced GridOp ops.
struct GridShapeFolder
    : OpRewritePatternWithSymbolTableCollection<GridShapeOp> {
  using OpRewritePatternWithSymbolTableCollection::
      OpRewritePatternWithSymbolTableCollection;
  LogicalResult matchAndRewrite(GridShapeOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    GridOp grid = symbolTableCollection.lookupNearestSymbolFrom<shard::GridOp>(
        op.getOperation(), op.getGridAttr());
    if (!grid) {
      return failure();
    }
    ArrayRef<GridAxis> opGridAxes = op.getAxes();
    SmallVector<GridAxis> opAxesIota;
    if (opGridAxes.empty()) {
      opAxesIota.resize(grid.getRank());
      std::iota(opAxesIota.begin(), opAxesIota.end(), 0);
      opGridAxes = opAxesIota;
    }
    if (llvm::all_of(opGridAxes, [&grid](GridAxis axis) {
          return ShapedType::isDynamic(grid.getShape()[axis]);
        })) {
      // All grid dimensions are dynamic. Nothing to fold.
      return failure();
    }

    SmallVector<Value> newResults(op->getResults().size());
    SmallVector<GridAxis> newShapeOpGridAxes;
    SmallVector<size_t> newToOldResultsIndexMap;

    for (size_t i = 0; i < opGridAxes.size(); ++i) {
      auto gridAxisSize = grid.getShape()[opGridAxes[i]];
      if (ShapedType::isDynamic(gridAxisSize)) {
        newToOldResultsIndexMap.push_back(i);
        newShapeOpGridAxes.push_back(opGridAxes[i]);
      } else {
        // Fold static grid axes.
        newResults[i] = arith::ConstantOp::create(
            builder, builder.getIndexAttr(gridAxisSize));
      }
    }

    // Leave only the dynamic grid axes to be queried.
    if (!newShapeOpGridAxes.empty()) {
      GridShapeOp newShapeOp =
          GridShapeOp::create(builder, grid.getSymName(), newShapeOpGridAxes);
      for (size_t i = 0; i < newShapeOp->getResults().size(); ++i) {
        newResults[newToOldResultsIndexMap[i]] = newShapeOp->getResults()[i];
      }
    }
    rewriter.replaceOp(op, newResults);

    return success();
  }
};

} // namespace

void populateFoldingPatterns(RewritePatternSet &patterns,
                             SymbolTableCollection &symbolTableCollection) {
  patterns.add<GridShapeFolder>(symbolTableCollection, patterns.getContext());
}

} // namespace shard
} // namespace mlir
