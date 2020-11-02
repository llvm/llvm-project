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
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
/// Merges subview operation with load/transferRead operation.
template <typename OpTy>
class LoadOpOfSubViewFolder final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy loadOp,
                                PatternRewriter &rewriter) const override;

private:
  void replaceOp(OpTy loadOp, SubViewOp subViewOp,
                 ArrayRef<Value> sourceIndices,
                 PatternRewriter &rewriter) const;
};

/// Merges subview operation with store/transferWriteOp operation.
template <typename OpTy>
class StoreOpOfSubViewFolder final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy storeOp,
                                PatternRewriter &rewriter) const override;

private:
  void replaceOp(OpTy StoreOp, SubViewOp subViewOp,
                 ArrayRef<Value> sourceIndices,
                 PatternRewriter &rewriter) const;
};

template <>
void LoadOpOfSubViewFolder<LoadOp>::replaceOp(LoadOp loadOp,
                                              SubViewOp subViewOp,
                                              ArrayRef<Value> sourceIndices,
                                              PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LoadOp>(loadOp, subViewOp.source(),
                                      sourceIndices);
}

template <>
void LoadOpOfSubViewFolder<vector::TransferReadOp>::replaceOp(
    vector::TransferReadOp loadOp, SubViewOp subViewOp,
    ArrayRef<Value> sourceIndices, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
      loadOp, loadOp.getVectorType(), subViewOp.source(), sourceIndices,
      loadOp.permutation_map(), loadOp.padding(), loadOp.maskedAttr());
}

template <>
void StoreOpOfSubViewFolder<StoreOp>::replaceOp(
    StoreOp storeOp, SubViewOp subViewOp, ArrayRef<Value> sourceIndices,
    PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<StoreOp>(storeOp, storeOp.value(),
                                       subViewOp.source(), sourceIndices);
}

template <>
void StoreOpOfSubViewFolder<vector::TransferWriteOp>::replaceOp(
    vector::TransferWriteOp tranferWriteOp, SubViewOp subViewOp,
    ArrayRef<Value> sourceIndices, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
      tranferWriteOp, tranferWriteOp.vector(), subViewOp.source(),
      sourceIndices, tranferWriteOp.permutation_map(),
      tranferWriteOp.maskedAttr());
}
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
// Folding SubViewOp and LoadOp/TransferReadOp.
//===----------------------------------------------------------------------===//

template <typename OpTy>
LogicalResult
LoadOpOfSubViewFolder<OpTy>::matchAndRewrite(OpTy loadOp,
                                             PatternRewriter &rewriter) const {
  auto subViewOp = loadOp.memref().template getDefiningOp<SubViewOp>();
  if (!subViewOp) {
    return failure();
  }
  SmallVector<Value, 4> sourceIndices;
  if (failed(resolveSourceIndices(loadOp.getLoc(), rewriter, subViewOp,
                                  loadOp.indices(), sourceIndices)))
    return failure();

  replaceOp(loadOp, subViewOp, sourceIndices, rewriter);
  return success();
}

//===----------------------------------------------------------------------===//
// Folding SubViewOp and StoreOp/TransferWriteOp.
//===----------------------------------------------------------------------===//

template <typename OpTy>
LogicalResult
StoreOpOfSubViewFolder<OpTy>::matchAndRewrite(OpTy storeOp,
                                              PatternRewriter &rewriter) const {
  auto subViewOp = storeOp.memref().template getDefiningOp<SubViewOp>();
  if (!subViewOp) {
    return failure();
  }
  SmallVector<Value, 4> sourceIndices;
  if (failed(resolveSourceIndices(storeOp.getLoc(), rewriter, subViewOp,
                                  storeOp.indices(), sourceIndices)))
    return failure();

  replaceOp(storeOp, subViewOp, sourceIndices, rewriter);
  return success();
}

//===----------------------------------------------------------------------===//
// Hook for adding patterns.
//===----------------------------------------------------------------------===//

void mlir::populateStdLegalizationPatternsForSPIRVLowering(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<LoadOpOfSubViewFolder<LoadOp>,
                  LoadOpOfSubViewFolder<vector::TransferReadOp>,
                  StoreOpOfSubViewFolder<StoreOp>,
                  StoreOpOfSubViewFolder<vector::TransferWriteOp>>(context);
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
  applyPatternsAndFoldGreedily(getOperation()->getRegions(),
                               std::move(patterns));
}

std::unique_ptr<Pass> mlir::createLegalizeStdOpsForSPIRVLoweringPass() {
  return std::make_unique<SPIRVLegalization>();
}
