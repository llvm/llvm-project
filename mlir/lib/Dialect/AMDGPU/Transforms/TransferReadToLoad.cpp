//===- TransferReadToLoad.cpp - Lowers masked transfer read to load -------===//
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
#include "mlir/Dialect/Arith/Utils/Utils.h"
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
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::amdgpu {
#define GEN_PASS_DEF_AMDGPUTRANSFERREADTOLOADPASS
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h.inc"
} // namespace mlir::amdgpu

using namespace mlir;
using namespace mlir::amdgpu;

/// This pattern supports lowering of:
/// `vector.transfer_read` to a combination of `vector.load`, `arith.select` and
/// `vector.broadcast` if all of the following hold:
/// - The transfer op is masked.
/// - The memref is in buffer address space.
/// - Stride of most minor memref dimension must be 1.
/// - Out-of-bounds masking is not required.
/// - If the memref's element type is a vector type then it coincides with the
///   result type.
/// - The permutation map doesn't perform permutation (broadcasting is allowed).
/// Note: those conditions mostly come from TransferReadToVectorLoadLowering
/// pass.
static LogicalResult transferPreconditions(
    PatternRewriter &rewriter, VectorTransferOpInterface xferOp,
    bool &requiresBroadcasting, VectorType &unbroadcastedVectorType) {
  if (!xferOp.getMask())
    return rewriter.notifyMatchFailure(xferOp, "Only support masked transfer");

  // Permutations are handled by VectorToSCF or
  // populateVectorTransferPermutationMapLoweringPatterns.
  // We let the 0-d corner case pass-through as it is supported.
  SmallVector<unsigned> broadcastedDims;
  if (!xferOp.getPermutationMap().isMinorIdentityWithBroadcasting(
          &broadcastedDims))
    return rewriter.notifyMatchFailure(xferOp, "not minor identity + bcast");

  auto memRefType = dyn_cast<MemRefType>(xferOp.getShapedType());
  if (!memRefType)
    return rewriter.notifyMatchFailure(xferOp, "not a memref source");

  Attribute addrSpace = memRefType.getMemorySpace();
  if (!addrSpace || !dyn_cast<amdgpu::AddressSpaceAttr>(addrSpace))
    return rewriter.notifyMatchFailure(xferOp, "no address space");

  if (dyn_cast<amdgpu::AddressSpaceAttr>(addrSpace).getValue() !=
      amdgpu::AddressSpace::FatRawBuffer)
    return rewriter.notifyMatchFailure(xferOp, "not in buffer address space");

  // Non-unit strides are handled by VectorToSCF.
  if (!memRefType.isLastDimUnitStride())
    return rewriter.notifyMatchFailure(xferOp, "!= 1 stride needs VectorToSCF");

  // If there is broadcasting involved then we first load the unbroadcasted
  // vector, and then broadcast it with `vector.broadcast`.
  ArrayRef<int64_t> vectorShape = xferOp.getVectorType().getShape();
  SmallVector<int64_t> unbroadcastedVectorShape(vectorShape);
  for (unsigned i : broadcastedDims)
    unbroadcastedVectorShape[i] = 1;
  unbroadcastedVectorType = xferOp.getVectorType().cloneWith(
      unbroadcastedVectorShape, xferOp.getVectorType().getElementType());
  requiresBroadcasting = !broadcastedDims.empty();

  // `vector.load` supports vector types as memref's elements only when the
  // resulting vector type is the same as the element type.
  auto memrefElTy = memRefType.getElementType();
  if (isa<VectorType>(memrefElTy) && memrefElTy != unbroadcastedVectorType)
    return rewriter.notifyMatchFailure(xferOp, "incompatible element type");

  // Otherwise, element types of the memref and the vector must match.
  if (!isa<VectorType>(memrefElTy) &&
      memrefElTy != xferOp.getVectorType().getElementType())
    return rewriter.notifyMatchFailure(xferOp, "non-matching element type");

  // Out-of-bounds dims are handled by MaterializeTransferMask.
  if (xferOp.hasOutOfBoundsDim())
    return rewriter.notifyMatchFailure(xferOp, "out-of-bounds needs mask");

  if (xferOp.getVectorType().getRank() != 1)
    // vector.maskedload operates on 1-D vectors.
    return rewriter.notifyMatchFailure(
        xferOp, "vector type is not rank 1, can't create masked load, needs "
                "VectorToSCF");

  return success();
}

static Value createVectorLoadForMaskedLoad(OpBuilder &builder, Location loc,
                                           vector::TransferReadOp readOp,
                                           bool requiresBroadcasting,
                                           VectorType unbroadcastedVectorType) {
  Value fill = builder.create<vector::SplatOp>(loc, unbroadcastedVectorType,
                                               readOp.getPadding());
  Value load = builder.create<vector::LoadOp>(
      loc, unbroadcastedVectorType, readOp.getSource(), readOp.getIndices());
  Value res = builder.create<arith::SelectOp>(loc, unbroadcastedVectorType,
                                              readOp.getMask(), load, fill);
  // Insert a broadcasting op if required.
  if (requiresBroadcasting) {
    res = builder.create<vector::BroadcastOp>(loc, readOp.getVectorType(), res);
  }
  return res;
}

namespace {

struct TransferReadLowering final : OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    if (readOp->hasAttr("amdgpu.transformed"))
      return failure();

    bool requiresBroadcasting = false;
    VectorType unbroadcastedVectorType;
    if (failed(transferPreconditions(rewriter, readOp, requiresBroadcasting,
                                     unbroadcastedVectorType))) {
      return failure();
    }

    Location loc = readOp.getLoc();
    Value src = readOp.getSource();

    VectorType vectorType = readOp.getVectorType();
    int64_t vectorSize = vectorType.getNumElements();
    int64_t elementBitWidth = vectorType.getElementTypeBitWidth();
    // Value linearIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<OpFoldResult> indices = readOp.getIndices();

    auto stridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, src);
    memref::LinearizedMemRefInfo linearizedInfo;
    OpFoldResult linearizedIndices;
    std::tie(linearizedInfo, linearizedIndices) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, elementBitWidth, elementBitWidth,
            stridedMetadata.getConstifiedMixedOffset(),
            stridedMetadata.getConstifiedMixedSizes(),
            stridedMetadata.getConstifiedMixedStrides(), indices);
    // OpFoldResult linearIndexSize = linearizedInfo.linearizedSize;
    Value linearIndex =
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices);

    // Note below doesn't give the correct result for the linearized size.
    // It compute the mutiplied sizes of all dimensions instead of taking
    // the maximum of each dimension size * stride.
    // TODO(jerryyin): Fix the getLinearizedMemRefOffsetAndSize() function
    // Value totalSize = getValueOrCreateConstantIndexOp(
    //    rewriter, loc, linearizedInfo.linearizedSize);
    SmallVector<AffineExpr> productExpressions;
    SmallVector<Value> productResults;
    unsigned sourceRank =
        cast<ShapedType>(readOp.getSource().getType()).getRank();

    SmallVector<AffineExpr> symbols(2 * sourceRank);
    SmallVector<Value> offsetValues(2 * sourceRank);
    bindSymbolsList(rewriter.getContext(), MutableArrayRef{symbols});
    for (size_t i = 0; i < sourceRank; ++i) {
      unsigned offsetIdx = 2 * i;
      productExpressions.push_back(symbols[offsetIdx] * symbols[offsetIdx + 1]);
      offsetValues[offsetIdx] = stridedMetadata.getStrides()[i];
      offsetValues[offsetIdx + 1] = stridedMetadata.getSizes()[i];
    }

    AffineMap maxMap = AffineMap::get(
        /*dimCount=*/0, /*symbolCount=*/symbols.size(), productExpressions,
        rewriter.getContext());
    Value totalSize =
        rewriter.create<affine::AffineMaxOp>(loc, maxMap, offsetValues);

    // delta = bufferSize - linearizedOffset
    Value vectorSizeOffset =
        rewriter.create<arith::ConstantIndexOp>(loc, vectorSize);
    Value delta = rewriter.create<arith::SubIOp>(loc, totalSize, linearIndex);

    // 1) check if delta < vectorSize
    Value isOutofBounds = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ule, delta, vectorSizeOffset);

    // 2) check if (detla(bytes) % (32 / elementBitwidth) != 0)
    Value deltaBytes = rewriter.create<arith::MulIOp>(
        loc, delta,
        rewriter.create<arith::ConstantIndexOp>(loc, elementBitWidth / 8));
    Value elementsPerWord = rewriter.create<arith::ConstantIndexOp>(
        loc, elementBitWidth < 32 ? 32 / elementBitWidth : 1);
    Value isNotWordAligned = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne,
        rewriter.create<arith::RemUIOp>(loc, deltaBytes, elementsPerWord),
        rewriter.create<arith::ConstantIndexOp>(loc, 0));

    // We take the fallback of transfer_read default lowering only it is both
    // out-of-bounds and not word aligned.
    Value ifCondition =
        rewriter.create<arith::AndIOp>(loc, isOutofBounds, isNotWordAligned);

    auto thenBuilder = [&](OpBuilder &builder, Location loc) {
      Operation *read = builder.clone(*readOp.getOperation());
      read->setAttr("amdgpu.transformed", builder.getUnitAttr());
      Value readResult = read->getResult(0);
      builder.create<scf::YieldOp>(loc, readResult);
    };

    auto elseBuilder = [&](OpBuilder &builder, Location loc) {
      Value res = createVectorLoadForMaskedLoad(
          builder, loc, readOp, requiresBroadcasting, unbroadcastedVectorType);
      rewriter.create<scf::YieldOp>(loc, res);
    };

    auto ifOp =
        rewriter.create<scf::IfOp>(loc, ifCondition, thenBuilder, elseBuilder);

    rewriter.replaceOp(readOp, ifOp);

    return success();
  }
};

} // namespace

void mlir::amdgpu::populateAmdgpuTransferReadToLoadPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TransferReadLowering>(patterns.getContext());
}

struct AmdgpuTransferReadToLoadPass final
    : amdgpu::impl::AmdgpuTransferReadToLoadPassBase<
          AmdgpuTransferReadToLoadPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateAmdgpuTransferReadToLoadPatterns(patterns);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
