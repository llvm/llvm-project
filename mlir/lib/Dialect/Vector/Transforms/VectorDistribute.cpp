//===- VectorDistribute.cpp - patterns to do vector distribution ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"

using namespace mlir;
using namespace mlir::vector;

static LogicalResult
rewriteWarpOpToScfFor(RewriterBase &rewriter, WarpExecuteOnLane0Op warpOp,
                      const WarpExecuteOnLane0LoweringOptions &options) {
  assert(warpOp.getBodyRegion().hasOneBlock() &&
         "expected WarpOp with single block");
  Block *warpOpBody = &warpOp.getBodyRegion().front();
  Location loc = warpOp.getLoc();

  // Passed all checks. Start rewriting.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(warpOp);

  // Create scf.if op.
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value isLane0 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 warpOp.getLaneid(), c0);
  auto ifOp = rewriter.create<scf::IfOp>(loc, isLane0,
                                         /*withElseRegion=*/false);
  rewriter.eraseOp(ifOp.thenBlock()->getTerminator());

  // Store vectors that are defined outside of warpOp into the scratch pad
  // buffer.
  SmallVector<Value> bbArgReplacements;
  for (const auto &it : llvm::enumerate(warpOp.getArgs())) {
    Value val = it.value();
    Value bbArg = warpOpBody->getArgument(it.index());

    rewriter.setInsertionPoint(ifOp);
    Value buffer = options.warpAllocationFn(warpOp->getLoc(), rewriter, warpOp,
                                            bbArg.getType());

    // Store arg vector into buffer.
    rewriter.setInsertionPoint(ifOp);
    auto vectorType = val.getType().cast<VectorType>();
    int64_t storeSize = vectorType.getShape()[0];
    Value storeOffset = rewriter.create<arith::MulIOp>(
        loc, warpOp.getLaneid(),
        rewriter.create<arith::ConstantIndexOp>(loc, storeSize));
    rewriter.create<vector::StoreOp>(loc, val, buffer, storeOffset);

    // Load bbArg vector from buffer.
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    auto bbArgType = bbArg.getType().cast<VectorType>();
    Value loadOp = rewriter.create<vector::LoadOp>(loc, bbArgType, buffer, c0);
    bbArgReplacements.push_back(loadOp);
  }

  // Insert sync after all the stores and before all the loads.
  if (!warpOp.getArgs().empty()) {
    rewriter.setInsertionPoint(ifOp);
    options.warpSyncronizationFn(warpOp->getLoc(), rewriter, warpOp);
  }

  // Move body of warpOp to ifOp.
  rewriter.mergeBlocks(warpOpBody, ifOp.thenBlock(), bbArgReplacements);

  // Rewrite terminator and compute replacements of WarpOp results.
  SmallVector<Value> replacements;
  auto yieldOp = cast<vector::YieldOp>(ifOp.thenBlock()->getTerminator());
  Location yieldLoc = yieldOp.getLoc();
  for (const auto &it : llvm::enumerate(yieldOp.operands())) {
    Value val = it.value();
    Type resultType = warpOp->getResultTypes()[it.index()];
    rewriter.setInsertionPoint(ifOp);
    Value buffer = options.warpAllocationFn(warpOp->getLoc(), rewriter, warpOp,
                                            val.getType());

    // Store yielded value into buffer.
    rewriter.setInsertionPoint(yieldOp);
    if (val.getType().isa<VectorType>())
      rewriter.create<vector::StoreOp>(yieldLoc, val, buffer, c0);
    else
      rewriter.create<memref::StoreOp>(yieldLoc, val, buffer, c0);

    // Load value from buffer (after warpOp).
    rewriter.setInsertionPointAfter(ifOp);
    if (resultType == val.getType()) {
      // Result type and yielded value type are the same. This is a broadcast.
      // E.g.:
      // %r = vector_ext.warp_execute_on_lane_0(...) -> (f32) {
      //   vector_ext.yield %cst : f32
      // }
      // Both types are f32. The constant %cst is broadcasted to all lanes.
      // This is described in more detail in the documentation of the op.
      Value loadOp = rewriter.create<memref::LoadOp>(loc, buffer, c0);
      replacements.push_back(loadOp);
    } else {
      auto loadedVectorType = resultType.cast<VectorType>();
      int64_t loadSize = loadedVectorType.getShape()[0];

      // loadOffset = laneid * loadSize
      Value loadOffset = rewriter.create<arith::MulIOp>(
          loc, warpOp.getLaneid(),
          rewriter.create<arith::ConstantIndexOp>(loc, loadSize));
      Value loadOp = rewriter.create<vector::LoadOp>(loc, loadedVectorType,
                                                     buffer, loadOffset);
      replacements.push_back(loadOp);
    }
  }

  // Insert sync after all the stores and before all the loads.
  if (!yieldOp.operands().empty()) {
    rewriter.setInsertionPointAfter(ifOp);
    options.warpSyncronizationFn(warpOp->getLoc(), rewriter, warpOp);
  }

  // Delete terminator and add empty scf.yield.
  rewriter.eraseOp(yieldOp);
  rewriter.setInsertionPointToEnd(ifOp.thenBlock());
  rewriter.create<scf::YieldOp>(yieldLoc);

  // Compute replacements for WarpOp results.
  rewriter.replaceOp(warpOp, replacements);

  return success();
}

namespace {

struct WarpOpToScfForPattern : public OpRewritePattern<WarpExecuteOnLane0Op> {
  WarpOpToScfForPattern(MLIRContext *context,
                        const WarpExecuteOnLane0LoweringOptions &options,
                        PatternBenefit benefit = 1)
      : OpRewritePattern<WarpExecuteOnLane0Op>(context, benefit),
        options(options) {}

  LogicalResult matchAndRewrite(WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    return rewriteWarpOpToScfFor(rewriter, warpOp, options);
  }

private:
  const WarpExecuteOnLane0LoweringOptions &options;
};

} // namespace

void mlir::vector::populateWarpExecuteOnLane0OpToScfForPattern(
    RewritePatternSet &patterns,
    const WarpExecuteOnLane0LoweringOptions &options) {
  patterns.add<WarpOpToScfForPattern>(patterns.getContext(), options);
}
