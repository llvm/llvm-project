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
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRVPass.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

/// Helpers to access the memref operand for each op.
static Value getMemRefOperand(memref::LoadOp op) { return op.memref(); }

static Value getMemRefOperand(vector::TransferReadOp op) { return op.source(); }

static Value getMemRefOperand(memref::StoreOp op) { return op.memref(); }

static Value getMemRefOperand(vector::TransferWriteOp op) {
  return op.source();
}

namespace {
/// Merges subview operation with load/transferRead operation.
template <typename OpTy>
class LoadOpOfSubViewFolder final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy loadOp,
                                PatternRewriter &rewriter) const override;

private:
  void replaceOp(OpTy loadOp, memref::SubViewOp subViewOp,
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
  void replaceOp(OpTy storeOp, memref::SubViewOp subViewOp,
                 ArrayRef<Value> sourceIndices,
                 PatternRewriter &rewriter) const;
};

template <>
void LoadOpOfSubViewFolder<memref::LoadOp>::replaceOp(
    memref::LoadOp loadOp, memref::SubViewOp subViewOp,
    ArrayRef<Value> sourceIndices, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<memref::LoadOp>(loadOp, subViewOp.source(),
                                              sourceIndices);
}

template <>
void LoadOpOfSubViewFolder<vector::TransferReadOp>::replaceOp(
    vector::TransferReadOp loadOp, memref::SubViewOp subViewOp,
    ArrayRef<Value> sourceIndices, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
      loadOp, loadOp.getVectorType(), subViewOp.source(), sourceIndices,
      loadOp.permutation_map(), loadOp.padding(), loadOp.in_boundsAttr());
}

template <>
void StoreOpOfSubViewFolder<memref::StoreOp>::replaceOp(
    memref::StoreOp storeOp, memref::SubViewOp subViewOp,
    ArrayRef<Value> sourceIndices, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<memref::StoreOp>(
      storeOp, storeOp.value(), subViewOp.source(), sourceIndices);
}

template <>
void StoreOpOfSubViewFolder<vector::TransferWriteOp>::replaceOp(
    vector::TransferWriteOp transferWriteOp, memref::SubViewOp subViewOp,
    ArrayRef<Value> sourceIndices, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
      transferWriteOp, transferWriteOp.vector(), subViewOp.source(),
      sourceIndices, transferWriteOp.permutation_map(),
      transferWriteOp.in_boundsAttr());
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
                     memref::SubViewOp subViewOp, ValueRange indices,
                     SmallVectorImpl<Value> &sourceIndices) {
  // TODO: Aborting when the offsets are static. There might be a way to fold
  // the subview op with load even if the offsets have been canonicalized
  // away.
  SmallVector<Range, 4> opRanges = subViewOp.getOrCreateRanges(rewriter, loc);
  auto opOffsets = llvm::map_range(opRanges, [](Range r) { return r.offset; });
  auto opStrides = llvm::map_range(opRanges, [](Range r) { return r.stride; });
  assert(opRanges.size() == indices.size() &&
         "expected as many indices as rank of subview op result type");

  // New indices for the load are the current indices * subview_stride +
  // subview_offset.
  sourceIndices.resize(indices.size());
  for (auto index : llvm::enumerate(indices)) {
    auto offset = *(opOffsets.begin() + index.index());
    auto stride = *(opStrides.begin() + index.index());
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
  auto subViewOp =
      getMemRefOperand(loadOp).template getDefiningOp<memref::SubViewOp>();
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
  auto subViewOp =
      getMemRefOperand(storeOp).template getDefiningOp<memref::SubViewOp>();
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
    RewritePatternSet &patterns) {
  patterns.add<LoadOpOfSubViewFolder<memref::LoadOp>,
               LoadOpOfSubViewFolder<vector::TransferReadOp>,
               StoreOpOfSubViewFolder<memref::StoreOp>,
               StoreOpOfSubViewFolder<vector::TransferWriteOp>>(
      patterns.getContext());
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
  RewritePatternSet patterns(&getContext());
  populateStdLegalizationPatternsForSPIRVLowering(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation()->getRegions(),
                                     std::move(patterns));
}

std::unique_ptr<Pass> mlir::createLegalizeStdOpsForSPIRVLoweringPass() {
  return std::make_unique<SPIRVLegalization>();
}
