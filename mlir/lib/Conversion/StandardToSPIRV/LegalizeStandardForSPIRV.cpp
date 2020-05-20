//===- LegalizeStandardForSPIRV.cpp - Legalize ops for SPIR-V lowering ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes operations before the conversion to SPIR-V
// dialect to handle ops that cannot be lowered directly.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;

namespace {
/// Merges subview operation with load operation.
class LoadOpOfSubViewFolder final : public OpRewritePattern<LoadOp> {
public:
  using OpRewritePattern<LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadOp loadOp,
                                PatternRewriter &rewriter) const override;
};

/// Merges subview operation with store operation.
class StoreOpOfSubViewFolder final : public OpRewritePattern<StoreOp> {
public:
  using OpRewritePattern<StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StoreOp storeOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Utility functions for op legalization.
//===----------------------------------------------------------------------===//

/// Given the 'indices' of an load/store operation where the memref is a result
/// of a subview op, returns the indices w.r.t to the source memref of the
/// subview op. For example
///
/// %0 = ... : memref<12x42xf32>
/// %1 = subview %0[%arg0, %arg1][][%stride1, %stride2] : memref<12x42xf32> to
///          memref<4x4xf32, offset=?, strides=[?, ?]>
/// %2 = load %1[%i1, %i2] : memref<4x4xf32, offset=?, strides=[?, ?]>
///
/// could be folded into
///
/// %2 = load %0[%arg0 + %i1 * %stride1][%arg1 + %i2 * %stride2] :
///          memref<12x42xf32>
static LogicalResult
resolveSourceIndices(Location loc, PatternRewriter &rewriter,
                     SubViewOp subViewOp, ValueRange indices,
                     SmallVectorImpl<Value> &sourceIndices) {
  // TODO: Aborting when the offsets are static. There might be a way to fold
  // the subview op with load even if the offsets have been canonicalized
  // away.
  SmallVector<Value, 4> opOffsets = subViewOp.getOrCreateOffsets(rewriter, loc);
  SmallVector<Value, 4> opStrides = subViewOp.getOrCreateStrides(rewriter, loc);
  assert(opOffsets.size() == indices.size() &&
         "expected as many indices as rank of subview op result type");
  assert(opStrides.size() == indices.size() &&
         "expected as many indices as rank of subview op result type");

  // New indices for the load are the current indices * subview_stride +
  // subview_offset.
  sourceIndices.resize(indices.size());
  for (auto index : llvm::enumerate(indices)) {
    auto offset = opOffsets[index.index()];
    auto stride = opStrides[index.index()];
    auto mul = rewriter.create<MulIOp>(loc, index.value(), stride);
    sourceIndices[index.index()] =
        rewriter.create<AddIOp>(loc, offset, mul).getResult();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Folding SubViewOp and LoadOp.
//===----------------------------------------------------------------------===//

LogicalResult
LoadOpOfSubViewFolder::matchAndRewrite(LoadOp loadOp,
                                       PatternRewriter &rewriter) const {
  auto subViewOp = loadOp.memref().getDefiningOp<SubViewOp>();
  if (!subViewOp) {
    return failure();
  }
  SmallVector<Value, 4> sourceIndices;
  if (failed(resolveSourceIndices(loadOp.getLoc(), rewriter, subViewOp,
                                  loadOp.indices(), sourceIndices)))
    return failure();

  rewriter.replaceOpWithNewOp<LoadOp>(loadOp, subViewOp.source(),
                                      sourceIndices);
  return success();
}

//===----------------------------------------------------------------------===//
// Folding SubViewOp and StoreOp.
//===----------------------------------------------------------------------===//

LogicalResult
StoreOpOfSubViewFolder::matchAndRewrite(StoreOp storeOp,
                                        PatternRewriter &rewriter) const {
  auto subViewOp = storeOp.memref().getDefiningOp<SubViewOp>();
  if (!subViewOp) {
    return failure();
  }
  SmallVector<Value, 4> sourceIndices;
  if (failed(resolveSourceIndices(storeOp.getLoc(), rewriter, subViewOp,
                                  storeOp.indices(), sourceIndices)))
    return failure();

  rewriter.replaceOpWithNewOp<StoreOp>(storeOp, storeOp.value(),
                                       subViewOp.source(), sourceIndices);
  return success();
}

//===----------------------------------------------------------------------===//
// Hook for adding patterns.
//===----------------------------------------------------------------------===//

void mlir::populateStdLegalizationPatternsForSPIRVLowering(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<LoadOpOfSubViewFolder, StoreOpOfSubViewFolder>(context);
}

//===----------------------------------------------------------------------===//
// Pass for testing just the legalization patterns.
//===----------------------------------------------------------------------===//

namespace {
struct SPIRVLegalization final
    : public LegalizeStandardForSPIRVBase<SPIRVLegalization> {
  void runOnOperation() override;
};
} // namespace

void SPIRVLegalization::runOnOperation() {
  OwningRewritePatternList patterns;
  auto *context = &getContext();
  populateStdLegalizationPatternsForSPIRVLowering(context, patterns);
  applyPatternsAndFoldGreedily(getOperation()->getRegions(), patterns);
}

std::unique_ptr<Pass> mlir::createLegalizeStdOpsForSPIRVLoweringPass() {
  return std::make_unique<SPIRVLegalization>();
}
