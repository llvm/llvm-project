//===----- FlattenMemRefs.cpp - MemRef ops flattener pass  ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns for flattening an multi-rank memref-related
// ops into 1-d memref ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_FLATTENMEMREFSPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

static Value getValueFromOpFoldResult(OpBuilder &rewriter, Location loc,
                                      OpFoldResult in) {
  if (Attribute offsetAttr = dyn_cast<Attribute>(in)) {
    return arith::ConstantIndexOp::create(
        rewriter, loc, cast<IntegerAttr>(offsetAttr).getInt());
  }
  return cast<Value>(in);
}

/// Returns a collapsed memref and the linearized index to access the element
/// at the specified indices.
static std::pair<Value, Value> getFlattenMemrefAndOffset(OpBuilder &rewriter,
                                                         Location loc,
                                                         Value source,
                                                         ValueRange indices) {
  int64_t sourceOffset;
  SmallVector<int64_t, 4> sourceStrides;
  auto sourceType = cast<MemRefType>(source.getType());
  if (failed(sourceType.getStridesAndOffset(sourceStrides, sourceOffset))) {
    assert(false);
  }

  memref::ExtractStridedMetadataOp stridedMetadata =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, source);

  auto typeBit = sourceType.getElementType().getIntOrFloatBitWidth();
  OpFoldResult linearizedIndices;
  memref::LinearizedMemRefInfo linearizedInfo;
  std::tie(linearizedInfo, linearizedIndices) =
      memref::getLinearizedMemRefOffsetAndSize(
          rewriter, loc, typeBit, typeBit,
          stridedMetadata.getConstifiedMixedOffset(),
          stridedMetadata.getConstifiedMixedSizes(),
          stridedMetadata.getConstifiedMixedStrides(),
          getAsOpFoldResult(indices));

  return std::make_pair(
      memref::ReinterpretCastOp::create(
          rewriter, loc, source,
          /*offset=*/linearizedInfo.linearizedOffset,
          /*sizes=*/
          ArrayRef<OpFoldResult>{linearizedInfo.linearizedSize},
          /*strides=*/
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)}),
      getValueFromOpFoldResult(rewriter, loc, linearizedIndices));
}

static bool needFlattening(Value val) {
  auto type = cast<MemRefType>(val.getType());
  return type.getRank() > 1;
}

static bool checkLayout(Value val) {
  auto type = cast<MemRefType>(val.getType());
  return type.getLayout().isIdentity() ||
         isa<StridedLayoutAttr>(type.getLayout());
}

namespace {
static bool hasSupportedElementType(Value memref) {
  auto type = cast<MemRefType>(memref.getType());
  return type.getElementType().isIntOrFloat();
}

/// Compute the type that will be used to linearize the memref.
/// Used so we don't create IR like `getLinearizedMemRefOffsetAndSize` would.
static FailureOr<MemRefType> getFlattenedMemRefType(MemRefType sourceType) {
  int64_t sourceOffset;
  SmallVector<int64_t> sourceStrides;
  if (failed(sourceType.getStridesAndOffset(sourceStrides, sourceOffset)))
    return failure();

  auto flatDimSize = SaturatedInteger::wrap(0);
  for (auto [size, stride] :
       llvm::zip_equal(sourceType.getShape(), sourceStrides)) {
    auto dimSize =
        SaturatedInteger::wrap(size) * SaturatedInteger::wrap(stride);
    flatDimSize = flatDimSize.smax(dimSize);
    if (flatDimSize.isSaturated())
      break;
  }

  if (sourceType.getLayout().isIdentity())
    return MemRefType::get(
        {flatDimSize.asInteger()}, sourceType.getElementType(),
        MemRefLayoutAttrInterface{}, sourceType.getMemorySpace());

  return MemRefType::get(
      {flatDimSize.asInteger()}, sourceType.getElementType(),
      StridedLayoutAttr::get(sourceType.getContext(), sourceOffset, {1}),
      sourceType.getMemorySpace());
}

/// Return whether `memref` has the basic properties needed for linearizing it
/// into a 1-D reinterpret_cast.
static LogicalResult checkFlattenableMemref(Operation *op, Value memref,
                                            PatternRewriter &rewriter) {
  if (!needFlattening(memref))
    return rewriter.notifyMatchFailure(op, "memref does not need flattening");
  if (!checkLayout(memref))
    return rewriter.notifyMatchFailure(op, "unsupported memref layout");
  if (!hasSupportedElementType(memref))
    return rewriter.notifyMatchFailure(op, "unsupported element type");
  return success();
}

/// Wrapeer around checking if the last memref dimension is contiguous that
/// provides nice failures message.
static LogicalResult hasUnitTrailingStride(Operation *op,
                                           TypedValue<MemRefType> memref,
                                           PatternRewriter &rewriter) {
  if (!memref.getType().areTrailingDimsContiguous(1))
    return rewriter.notifyMatchFailure(
        op, "cannot preserve non-unit trailing access stride");

  return success();
}

static LogicalResult
canLinearizeAccessedShape(memref::IndexedAccessOpInterface op,
                          TypedValue<MemRefType> memref,
                          PatternRewriter &rewriter) {
  SmallVector<int64_t> accessedShape = op.getAccessedShape();
  if (accessedShape.empty())
    return success();
  if (accessedShape.size() > 1)
    return rewriter.notifyMatchFailure(
        op, "cannot preserve multi-dimensional accessed shape");

  return hasUnitTrailingStride(op, memref, rewriter);
}

static LogicalResult canFlattenTransferOp(VectorTransferOpInterface op,
                                          TypedValue<MemRefType> memref,
                                          PatternRewriter &rewriter) {
  // For vector.transfer_read/write, must make sure:
  // 1. all accesses are inbounds,
  // 2. has a minor identity permutation map, and
  // 3. has at most one transfer dimension.
  AffineMap permutationMap = op.getPermutationMap();
  if (!permutationMap.isMinorIdentity())
    return rewriter.notifyMatchFailure(
        op, "only identity or minor identity permutation map is supported");

  if (op.hasOutOfBoundsDim())
    return rewriter.notifyMatchFailure(op, "only inbounds are supported");

  if (op.getTransferRank() > 1)
    return rewriter.notifyMatchFailure(
        op, "cannot flatten multi-dimensional vector transfer");

  if (op.getTransferRank() > 0 &&
      failed(hasUnitTrailingStride(op, memref, rewriter)))
    return failure();

  return success();
}

// Pattern for memref::AllocOp and memref::AllocaOp.
//
// The "source" memref for these ops is the op's own result, so the generic
// indexed access pattern cannot be used: getFlattenMemrefAndOffset would
// insert ExtractStridedMetadataOp and ReinterpretCastOp that use op.result
// before this op in the block. After replaceOpWithNewOp the original result is
// RAUW'd to the new ReinterpretCastOp, leaving the earlier ops with forward
// references (domination violations) caught by
// MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS.
//
// Instead, sizes and strides are computed from the op's operands and type
// (which all dominate the op), avoiding any reference to op.result until the
// final replaceOpWithNewOp.
template <typename AllocLikeOp>
struct AllocLikeFlattenPattern final : public OpRewritePattern<AllocLikeOp> {
  using Base = OpRewritePattern<AllocLikeOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(AllocLikeOp op,
                                PatternRewriter &rewriter) const override {
    if (!needFlattening(op.getMemref()) || !checkLayout(op.getMemref()))
      return failure();

    Location loc = op->getLoc();
    auto memrefType = cast<MemRefType>(op.getType());
    auto elemType = memrefType.getElementType();
    if (!elemType.isIntOrFloat())
      return failure();
    unsigned elemBitWidth = elemType.getIntOrFloatBitWidth();

    SmallVector<OpFoldResult> sizes = op.getMixedSizes();

    int64_t staticOffset;
    SmallVector<int64_t> staticStrides;
    if (failed(memrefType.getStridesAndOffset(staticStrides, staticOffset)))
      return failure();
    if (staticOffset == ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(op, "dynamic offset not supported");
    SmallVector<OpFoldResult> strides;
    strides.reserve(staticStrides.size());
    for (int64_t stride : staticStrides) {
      if (stride == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(op,
                                           "dynamic stride cannot be computed");
      strides.push_back(rewriter.getIndexAttr(stride));
    }

    // Compute the linearized flat extent from sizes and strides (no SSA ops
    // referencing op.result are created here).
    memref::LinearizedMemRefInfo linearizedInfo;
    OpFoldResult linearizedOffset;
    std::tie(linearizedInfo, linearizedOffset) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, elemBitWidth, elemBitWidth, rewriter.getIndexAttr(0),
            sizes, strides);
    (void)linearizedOffset;

    // The total allocation must cover [0, staticOffset + linearizedExtent).
    // When the offset is non-zero, add it to the computed extent so that the
    // buffer is large enough for elements accessed at positions
    // [staticOffset, staticOffset + linearizedExtent).
    OpFoldResult flatSizeOfr = linearizedInfo.linearizedSize;
    if (staticOffset != 0) {
      AffineExpr s0;
      bindSymbols(rewriter.getContext(), s0);
      flatSizeOfr = affine::makeComposedFoldedAffineApply(
          rewriter, loc, s0 + staticOffset, {flatSizeOfr});
    }

    // Build the flat 1-D MemRefType. The linearized size may be static or
    // dynamic (OpFoldResult of either IntegerAttr or a Value).
    int64_t flatDimSize = ShapedType::kDynamic;
    if (auto attr = dyn_cast<Attribute>(flatSizeOfr))
      if (auto intAttr = dyn_cast<IntegerAttr>(attr))
        flatDimSize = intAttr.getInt();

    auto flatMemrefType =
        MemRefType::get({flatDimSize}, memrefType.getElementType(),
                        StridedLayoutAttr::get(rewriter.getContext(), 0, {1}),
                        memrefType.getMemorySpace());

    // Collect the flat dynamic-size operand (empty for fully-static case).
    SmallVector<Value, 1> dynSizes;
    if (flatDimSize == ShapedType::kDynamic)
      dynSizes.push_back(getValueFromOpFoldResult(rewriter, loc, flatSizeOfr));

    auto newOp = AllocLikeOp::create(rewriter, loc, flatMemrefType, dynSizes,
                                     op.getAlignmentAttr());
    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, cast<MemRefType>(op.getType()), newOp,
        rewriter.getIndexAttr(staticOffset), sizes, strides);
    return success();
  }
};

/// Pattern that flattens any IndexedAccessOpInterface op.
struct IndexedAccessOpFlattenPattern final
    : public OpInterfaceRewritePattern<memref::IndexedAccessOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(memref::IndexedAccessOpInterface op,
                                PatternRewriter &rewriter) const override {
    TypedValue<MemRefType> memref = op.getAccessedMemref();
    if (!memref)
      return rewriter.notifyMatchFailure(op, "not accessing a memref");
    if (failed(checkFlattenableMemref(op, memref, rewriter)))
      return failure();
    if (failed(canLinearizeAccessedShape(op, memref, rewriter)))
      return failure();

    auto [flatMemref, offset] = getFlattenMemrefAndOffset(
        rewriter, op->getLoc(), memref, op.getIndices());
    std::optional<SmallVector<Value>> replacementValues =
        op.updateMemrefAndIndices(rewriter, flatMemref, ValueRange{offset});
    if (replacementValues)
      rewriter.replaceOp(op, *replacementValues);
    return success();
  }
};

/// Flatten operations that use VectorTransferOpInterface. Transfer ops have
/// permutation-map and in_bounds semantics that are separate from
/// IndexedAccessOpInterface, so use updateStartingPosition directly.
struct VectorTransferOpFlattenPattern final
    : public OpInterfaceRewritePattern<VectorTransferOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(VectorTransferOpInterface op,
                                PatternRewriter &rewriter) const override {
    auto memref = dyn_cast<TypedValue<MemRefType>>(op.getBase());
    if (!memref)
      return rewriter.notifyMatchFailure(op, "not accessing a memref");
    if (failed(checkFlattenableMemref(op, memref, rewriter)))
      return failure();
    if (failed(canFlattenTransferOp(op, memref, rewriter)))
      return failure();

    FailureOr<MemRefType> flatMemrefType =
        getFlattenedMemRefType(memref.getType());
    if (failed(flatMemrefType))
      return failure();
    AffineMap newPermutationMap = AffineMap::getMinorIdentityMap(
        /*dims=*/1, op.getTransferRank(), op.getContext());
    if (failed(
            op.mayUpdateStartingPosition(*flatMemrefType, newPermutationMap)))
      return rewriter.notifyMatchFailure(op,
                                         "failed op-specific preconditions");

    auto [flatMemref, offset] = getFlattenMemrefAndOffset(
        rewriter, op->getLoc(), memref, op.getIndices());
    op.updateStartingPosition(rewriter, flatMemref, ValueRange{offset},
                              AffineMapAttr::get(newPermutationMap));
    return success();
  }
};

/// Flatten the source and destination memref/index pairs of indexed memcpy-like
/// operations such as memref.dma_start.
struct FlattenedMemrefAccess {
  Value memref;
  Value index;
};

/// Flatten all IndexedMemCopyOpInterface operations.
struct IndexedMemCopyOpFlattenPattern final
    : public OpInterfaceRewritePattern<memref::IndexedMemCopyOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(memref::IndexedMemCopyOpInterface op,
                                PatternRewriter &rewriter) const override {
    TypedValue<MemRefType> src = op.getSrc();
    TypedValue<MemRefType> dst = op.getDst();
    if (!src && !dst)
      return rewriter.notifyMatchFailure(op, "not copying between memrefs");

    auto tryFlatten =
        [&](TypedValue<MemRefType> memref,
            ValueRange indices) -> std::optional<FlattenedMemrefAccess> {
      if (!memref || !needFlattening(memref))
        return std::nullopt;
      if (failed(checkFlattenableMemref(op, memref, rewriter)))
        return std::nullopt;

      auto [flatMemref, offset] =
          getFlattenMemrefAndOffset(rewriter, op->getLoc(), memref, indices);
      return FlattenedMemrefAccess{flatMemref, offset};
    };

    std::optional<FlattenedMemrefAccess> newSrc =
        tryFlatten(src, op.getSrcIndices());
    std::optional<FlattenedMemrefAccess> newDst =
        tryFlatten(dst, op.getDstIndices());
    if (!newSrc && !newDst)
      return rewriter.notifyMatchFailure(
          op, "no source or destination memref needed flattening");

    Value srcMemref = src;
    ValueRange srcIndices = op.getSrcIndices();
    if (newSrc) {
      srcMemref = newSrc->memref;
      srcIndices = ValueRange(newSrc->index);
    }

    Value dstMemref = dst;
    ValueRange dstIndices = op.getDstIndices();
    if (newDst) {
      dstMemref = newDst->memref;
      dstIndices = ValueRange(newDst->index);
    }

    op.setMemrefsAndIndices(rewriter, srcMemref, srcIndices, dstMemref,
                            dstIndices);
    return success();
  }
};

struct FlattenMemrefsPass
    : public mlir::memref::impl::FlattenMemrefsPassBase<FlattenMemrefsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    memref::populateFlattenMemrefsPatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void memref::populateFlattenMemrefsPatterns(RewritePatternSet &patterns) {
  patterns.insert<IndexedAccessOpFlattenPattern, IndexedMemCopyOpFlattenPattern,
                  VectorTransferOpFlattenPattern,
                  AllocLikeFlattenPattern<memref::AllocOp>,
                  AllocLikeFlattenPattern<memref::AllocaOp>>(
      patterns.getContext());
}
