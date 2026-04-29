//===- Simplify.cpp - Shard Simplify ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Shard/Transforms/Simplify.h"
#include "TransformsDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Shard/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <numeric>

namespace mlir {
namespace shard {

#define GEN_PASS_DEF_SHARDSIMPLIFY
#include "mlir/Dialect/Shard/Transforms/Passes.h.inc"

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

// Simplify AllSliceOp(AllReduceOp) -> ReduceScatterOp when both ops share the
// same grid and grid_axes.
//
// AllReduceOp performs an element-wise reduction across all devices in the
// group, and AllSliceOp then slices (scatters) the result along a tensor
// dimension. This is exactly what ReduceScatterOp does in a single collective.
//
// With a ring algorithm over N ranks and M elements:
//   AllReduce:      2*(N-1) steps of M/N each  =>  ~2M total data transferred
//   AllSlice:       local slice, no communication
//   ReduceScatter:  (N-1) steps of M/N each    =>  ~M total data transferred
// So this fusion roughly halves the communication volume.
//
// Memory-wise, AllReduce produces a full-sized M-element result that the
// subsequent AllSlice must keep alive until the slice is taken. ReduceScatter
// only materializes the M/N-element local slice, reducing peak memory by
// a factor of N.
struct AllReduceAllSliceSimplification : OpRewritePattern<AllSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    // Check if the input to AllSliceOp is produced by an AllReduceOp.
    auto reduceOp = sliceOp.getInput().getDefiningOp<AllReduceOp>();
    if (!reduceOp || !reduceOp->hasOneUse())
      return failure();

    // Both ops must operate on the same grid and grid axes.
    if (reduceOp.getGrid() != sliceOp.getGrid() ||
        reduceOp.getGridAxes() != sliceOp.getGridAxes())
      return failure();

    // Replace with a single ReduceScatterOp.
    rewriter.replaceOpWithNewOp<ReduceScatterOp>(
        sliceOp, sliceOp.getResult().getType(), sliceOp.getGridAttr(),
        sliceOp.getGridAxesAttr(), reduceOp.getInput(),
        reduceOp.getReductionAttr(), sliceOp.getSliceAxisAttr());

    return success();
  }
};

} // namespace

void populateSimplifyPatterns(RewritePatternSet &patterns,
                              SymbolTableCollection &symbolTableCollection) {
  populateAllReduceEndomorphismSimplifyPatterns<arith::AddFOp>(
      patterns, ReductionKind::Sum);
  populateAllReduceEndomorphismSimplifyPatterns<arith::AddIOp>(
      patterns, ReductionKind::Sum);

  populateAllReduceEndomorphismSimplifyPatterns<arith::MinimumFOp>(
      patterns, ReductionKind::Min);
  populateAllReduceEndomorphismSimplifyPatterns<arith::MinSIOp>(
      patterns, ReductionKind::Min);
  populateAllReduceEndomorphismSimplifyPatterns<arith::MinUIOp>(
      patterns, ReductionKind::Min);

  populateAllReduceEndomorphismSimplifyPatterns<arith::MaximumFOp>(
      patterns, ReductionKind::Max);
  populateAllReduceEndomorphismSimplifyPatterns<arith::MaxSIOp>(
      patterns, ReductionKind::Max);
  populateAllReduceEndomorphismSimplifyPatterns<arith::MaxUIOp>(
      patterns, ReductionKind::Max);

  patterns.add<AllReduceAllSliceSimplification>(patterns.getContext());

  // TODO: add simplify patterns for all-gather and other collectives.

  populateFoldingPatterns(patterns, symbolTableCollection);
}

void populateFoldingPatterns(RewritePatternSet &patterns,
                             SymbolTableCollection &symbolTableCollection) {
  patterns.add<GridShapeFolder>(symbolTableCollection, patterns.getContext());
}

namespace {

struct ShardSimplifyPass : public impl::ShardSimplifyBase<ShardSimplifyPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    SymbolTableCollection symbolTableCollection;
    populateSimplifyPatterns(patterns, symbolTableCollection);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace shard
} // namespace mlir
