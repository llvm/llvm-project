//===- TransferReadToLoad.cpp - Lowers masked transfer read to load -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
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

namespace {

struct TransferReadLowering final : OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {

    bool requiresBroadcasting = false;
    VectorType unbroadcastedVectorType;
    if (failed(transferPreconditions(rewriter, readOp, requiresBroadcasting,
                                     unbroadcastedVectorType))) {
      return failure();
    }

    Location loc = readOp.getLoc();
    Value fill = rewriter.create<vector::SplatOp>(loc, unbroadcastedVectorType,
                                                  readOp.getPadding());
    Value load = rewriter.create<vector::LoadOp>(
        loc, unbroadcastedVectorType, readOp.getSource(), readOp.getIndices());
    Value res = rewriter.create<arith::SelectOp>(loc, unbroadcastedVectorType,
                                                 readOp.getMask(), load, fill);

    // Insert a broadcasting op if required.
    if (requiresBroadcasting) {
      res = rewriter.create<vector::BroadcastOp>(loc, readOp.getVectorType(),
                                                 res);
    }

    rewriter.replaceOp(readOp, res);

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
