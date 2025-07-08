//===- MaskedloadToLoad.cpp - Lowers maskedload to load -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::amdgpu {
#define GEN_PASS_DEF_AMDGPUMASKEDLOADTOLOADPASS
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h.inc"
} // namespace mlir::amdgpu

using namespace mlir;
using namespace mlir::amdgpu;

/// This pattern supports lowering of: `vector.maskedload` to `vector.load`
/// and `arith.select` if the memref is in buffer address space.
static LogicalResult baseInBufferAddrSpace(PatternRewriter &rewriter,
                                           vector::MaskedLoadOp maskedOp) {
  auto memRefType = dyn_cast<MemRefType>(maskedOp.getBase().getType());
  if (!memRefType)
    return rewriter.notifyMatchFailure(maskedOp, "not a memref source");

  Attribute addrSpace = memRefType.getMemorySpace();
  if (!isa_and_nonnull<amdgpu::AddressSpaceAttr>(addrSpace))
    return rewriter.notifyMatchFailure(maskedOp, "no address space");

  if (dyn_cast<amdgpu::AddressSpaceAttr>(addrSpace).getValue() !=
      amdgpu::AddressSpace::FatRawBuffer)
    return rewriter.notifyMatchFailure(maskedOp, "not in buffer address space");

  return success();
}

static Value createVectorLoadForMaskedLoad(OpBuilder &builder, Location loc,
                                           vector::MaskedLoadOp maskedOp) {
  VectorType vectorType = maskedOp.getVectorType();
  Value load = builder.create<vector::LoadOp>(
      loc, vectorType, maskedOp.getBase(), maskedOp.getIndices());
  Value res = builder.create<arith::SelectOp>(
      loc, vectorType, maskedOp.getMask(), load, maskedOp.getPassThru());
  return res;
}

static constexpr char kMaskedloadNeedsMask[] =
    "amdgpu.buffer_maskedload_needs_mask";

namespace {

struct MaskedLoadLowering final : OpRewritePattern<vector::MaskedLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskedLoadOp maskedOp,
                                PatternRewriter &rewriter) const override {
    if (maskedOp->hasAttr(kMaskedloadNeedsMask))
      return failure();

    if (failed(baseInBufferAddrSpace(rewriter, maskedOp))) {
      return failure();
    }

    Location loc = maskedOp.getLoc();
    Value src = maskedOp.getBase();

    VectorType vectorType = maskedOp.getVectorType();
    int64_t vectorSize = vectorType.getNumElements();
    int64_t elementBitWidth = vectorType.getElementTypeBitWidth();
    SmallVector<OpFoldResult> indices = maskedOp.getIndices();

    auto stridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, src);
    SmallVector<OpFoldResult> strides =
        stridedMetadata.getConstifiedMixedStrides();
    SmallVector<OpFoldResult> sizes = stridedMetadata.getConstifiedMixedSizes();
    OpFoldResult offset = stridedMetadata.getConstifiedMixedOffset();
    memref::LinearizedMemRefInfo linearizedInfo;
    OpFoldResult linearizedIndices;
    std::tie(linearizedInfo, linearizedIndices) =
        memref::getLinearizedMemRefOffsetAndSize(rewriter, loc, elementBitWidth,
                                                 elementBitWidth, offset, sizes,
                                                 strides, indices);

    // delta = bufferSize - linearizedOffset
    Value vectorSizeOffset =
        rewriter.create<arith::ConstantIndexOp>(loc, vectorSize);
    Value linearIndex =
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices);
    Value totalSize = getValueOrCreateConstantIndexOp(
        rewriter, loc, linearizedInfo.linearizedSize);
    Value delta = rewriter.create<arith::SubIOp>(loc, totalSize, linearIndex);

    // 1) check if delta < vectorSize
    Value isOutofBounds = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, delta, vectorSizeOffset);

    // 2) check if (detla % elements_per_word != 0)
    Value elementsPerWord = rewriter.create<arith::ConstantIndexOp>(
        loc, llvm::divideCeil(32, elementBitWidth));
    Value isNotWordAligned = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne,
        rewriter.create<arith::RemUIOp>(loc, delta, elementsPerWord),
        rewriter.create<arith::ConstantIndexOp>(loc, 0));

    // We take the fallback of maskedload default lowering only it is both
    // out-of-bounds and not word aligned. The fallback ensures correct results
    // when loading at the boundary of the buffer since buffer load returns
    // inconsistent zeros for the whole word when boundary is crossed.
    Value ifCondition =
        rewriter.create<arith::AndIOp>(loc, isOutofBounds, isNotWordAligned);

    auto thenBuilder = [&](OpBuilder &builder, Location loc) {
      Operation *read = builder.clone(*maskedOp.getOperation());
      read->setAttr(kMaskedloadNeedsMask, builder.getUnitAttr());
      Value readResult = read->getResult(0);
      builder.create<scf::YieldOp>(loc, readResult);
    };

    auto elseBuilder = [&](OpBuilder &builder, Location loc) {
      Value res = createVectorLoadForMaskedLoad(builder, loc, maskedOp);
      rewriter.create<scf::YieldOp>(loc, res);
    };

    auto ifOp =
        rewriter.create<scf::IfOp>(loc, ifCondition, thenBuilder, elseBuilder);

    rewriter.replaceOp(maskedOp, ifOp);

    return success();
  }
};

} // namespace

void mlir::amdgpu::populateAmdgpuMaskedloadToLoadPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<MaskedLoadLowering>(patterns.getContext(), benefit);
}

struct AmdgpuMaskedloadToLoadPass final
    : amdgpu::impl::AmdgpuMaskedloadToLoadPassBase<AmdgpuMaskedloadToLoadPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateAmdgpuMaskedloadToLoadPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
