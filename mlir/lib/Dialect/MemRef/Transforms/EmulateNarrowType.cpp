//===- EmulateNarrowType.cpp - Narrow type emulation ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <type_traits>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Converts a memref::ReinterpretCastOp to the converted type. The result
/// MemRefType of the old op must have a rank and stride of 1, with static
/// offset and size. The number of bits in the offset must evenly divide the
/// bitwidth of the new converted type.
static LogicalResult
convertCastingOp(ConversionPatternRewriter &rewriter,
                 memref::ReinterpretCastOp::Adaptor adaptor,
                 memref::ReinterpretCastOp op, MemRefType newTy) {
  auto convertedElementType = newTy.getElementType();
  auto oldElementType = op.getType().getElementType();
  int srcBits = oldElementType.getIntOrFloatBitWidth();
  int dstBits = convertedElementType.getIntOrFloatBitWidth();
  if (dstBits % srcBits != 0) {
    return rewriter.notifyMatchFailure(op,
                                       "only dstBits % srcBits == 0 supported");
  }

  // Only support stride of 1.
  if (llvm::any_of(op.getStaticStrides(),
                   [](int64_t stride) { return stride != 1; })) {
    return rewriter.notifyMatchFailure(op->getLoc(),
                                       "stride != 1 is not supported");
  }

  auto sizes = op.getStaticSizes();
  int64_t offset = op.getStaticOffset(0);
  // Only support static sizes and offsets.
  if (llvm::is_contained(sizes, ShapedType::kDynamic) ||
      offset == ShapedType::kDynamic) {
    return rewriter.notifyMatchFailure(
        op, "dynamic size or offset is not supported");
  }

  int elementsPerByte = dstBits / srcBits;
  if (offset % elementsPerByte != 0) {
    return rewriter.notifyMatchFailure(
        op, "offset not multiple of elementsPerByte is not supported");
  }

  SmallVector<int64_t> size;
  if (!sizes.empty())
    size.push_back(llvm::divideCeilSigned(sizes[0], elementsPerByte));
  offset = offset / elementsPerByte;

  rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
      op, newTy, adaptor.getSource(), offset, size, op.getStaticStrides());
  return success();
}

/// When data is loaded/stored in `targetBits` granularity, but is used in
/// `sourceBits` granularity (`sourceBits` < `targetBits`), the `targetBits` is
/// treated as an array of elements of width `sourceBits`.
/// Return the bit offset of the value at position `srcIdx`. For example, if
/// `sourceBits` equals to 4 and `targetBits` equals to 8, the x-th element is
/// located at (x % 2) * 4. Because there are two elements in one i8, and one
/// element has 4 bits.
static Value getOffsetForBitwidth(Location loc, OpFoldResult srcIdx,
                                  int sourceBits, int targetBits,
                                  OpBuilder &builder) {
  assert(targetBits % sourceBits == 0);
  AffineExpr s0;
  bindSymbols(builder.getContext(), s0);
  int scaleFactor = targetBits / sourceBits;
  AffineExpr offsetExpr = (s0 % scaleFactor) * sourceBits;
  OpFoldResult offsetVal =
      affine::makeComposedFoldedAffineApply(builder, loc, offsetExpr, {srcIdx});
  Value bitOffset = getValueOrCreateConstantIndexOp(builder, loc, offsetVal);
  IntegerType dstType = builder.getIntegerType(targetBits);
  return arith::IndexCastOp::create(builder, loc, dstType, bitOffset);
}

/// When writing a subbyte size, masked bitwise operations are used to only
/// modify the relevant bits. This function returns an and mask for clearing
/// the destination bits in a subbyte write. E.g., when writing to the second
/// i4 in an i32, 0xFFFFFF0F is created.
static Value getSubByteWriteMask(Location loc, OpFoldResult linearizedIndices,
                                 int64_t srcBits, int64_t dstBits,
                                 Value bitwidthOffset, OpBuilder &builder) {
  auto dstIntegerType = builder.getIntegerType(dstBits);
  auto maskRightAlignedAttr =
      builder.getIntegerAttr(dstIntegerType, (1 << srcBits) - 1);
  Value maskRightAligned = arith::ConstantOp::create(
      builder, loc, dstIntegerType, maskRightAlignedAttr);
  Value writeMaskInverse =
      arith::ShLIOp::create(builder, loc, maskRightAligned, bitwidthOffset);
  auto flipValAttr = builder.getIntegerAttr(dstIntegerType, -1);
  Value flipVal =
      arith::ConstantOp::create(builder, loc, dstIntegerType, flipValAttr);
  return arith::XOrIOp::create(builder, loc, writeMaskInverse, flipVal);
}

/// Returns the scaled linearized index based on the `srcBits` and `dstBits`
/// sizes. The input `linearizedIndex` has the granularity of `srcBits`, and
/// the returned index has the granularity of `dstBits`
static Value getIndicesForLoadOrStore(OpBuilder &builder, Location loc,
                                      OpFoldResult linearizedIndex,
                                      int64_t srcBits, int64_t dstBits) {
  AffineExpr s0;
  bindSymbols(builder.getContext(), s0);
  int64_t scaler = dstBits / srcBits;
  OpFoldResult scaledLinearizedIndices = affine::makeComposedFoldedAffineApply(
      builder, loc, s0.floorDiv(scaler), {linearizedIndex});
  return getValueOrCreateConstantIndexOp(builder, loc, scaledLinearizedIndices);
}

static OpFoldResult
getLinearizedSrcIndices(OpBuilder &builder, Location loc, int64_t srcBits,
                        const SmallVector<OpFoldResult> &indices,
                        Value memref) {
  auto stridedMetadata =
      memref::ExtractStridedMetadataOp::create(builder, loc, memref);
  OpFoldResult linearizedIndices;
  std::tie(std::ignore, linearizedIndices) =
      memref::getLinearizedMemRefOffsetAndSize(
          builder, loc, srcBits, srcBits,
          stridedMetadata.getConstifiedMixedOffset(),
          stridedMetadata.getConstifiedMixedSizes(),
          stridedMetadata.getConstifiedMixedStrides(), indices);
  return linearizedIndices;
}

namespace {

//===----------------------------------------------------------------------===//
// ConvertMemRefAllocation
//===----------------------------------------------------------------------===//

template <typename OpTy>
struct ConvertMemRefAllocation final : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static_assert(std::is_same<OpTy, memref::AllocOp>() ||
                      std::is_same<OpTy, memref::AllocaOp>(),
                  "expected only memref::AllocOp or memref::AllocaOp");
    auto currentType = cast<MemRefType>(op.getMemref().getType());
    auto newResultType =
        this->getTypeConverter()->template convertType<MemRefType>(
            op.getType());
    if (!newResultType) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to convert memref type: {0}", op.getType()));
    }

    // Special case zero-rank memrefs.
    if (currentType.getRank() == 0) {
      rewriter.replaceOpWithNewOp<OpTy>(op, newResultType, ValueRange{},
                                        adaptor.getSymbolOperands(),
                                        adaptor.getAlignmentAttr());
      return success();
    }

    Location loc = op.getLoc();
    OpFoldResult zero = rewriter.getIndexAttr(0);

    // Get linearized type.
    int srcBits = currentType.getElementType().getIntOrFloatBitWidth();
    int dstBits = newResultType.getElementType().getIntOrFloatBitWidth();
    SmallVector<OpFoldResult> sizes = op.getMixedSizes();

    memref::LinearizedMemRefInfo linearizedMemRefInfo =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, srcBits, dstBits, /*offset =*/zero, sizes);
    SmallVector<Value> dynamicLinearizedSize;
    if (!newResultType.hasStaticShape()) {
      dynamicLinearizedSize.push_back(getValueOrCreateConstantIndexOp(
          rewriter, loc, linearizedMemRefInfo.linearizedSize));
    }

    rewriter.replaceOpWithNewOp<OpTy>(op, newResultType, dynamicLinearizedSize,
                                      adaptor.getSymbolOperands(),
                                      adaptor.getAlignmentAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMemRefAssumeAlignment
//===----------------------------------------------------------------------===//

struct ConvertMemRefAssumeAlignment final
    : OpConversionPattern<memref::AssumeAlignmentOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AssumeAlignmentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getMemref().getType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to convert memref type: {0}",
                                      op.getMemref().getType()));
    }

    rewriter.replaceOpWithNewOp<memref::AssumeAlignmentOp>(
        op, newTy, adaptor.getMemref(), adaptor.getAlignmentAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMemRefCopy
//===----------------------------------------------------------------------===//

struct ConvertMemRefCopy final : OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto maybeRankedSource = dyn_cast<MemRefType>(op.getSource().getType());
    auto maybeRankedDest = dyn_cast<MemRefType>(op.getTarget().getType());
    if (maybeRankedSource && maybeRankedDest &&
        maybeRankedSource.getLayout() != maybeRankedDest.getLayout())
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("memref.copy emulation with distinct layouts ({0} "
                            "and {1}) is currently unimplemented",
                            maybeRankedSource.getLayout(),
                            maybeRankedDest.getLayout()));
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, adaptor.getSource(),
                                                adaptor.getTarget());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMemRefDealloc
//===----------------------------------------------------------------------===//

struct ConvertMemRefDealloc final : OpConversionPattern<memref::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::DeallocOp>(op, adaptor.getMemref());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMemRefLoad
//===----------------------------------------------------------------------===//

struct ConvertMemRefLoad final : OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto convertedType = cast<MemRefType>(adaptor.getMemref().getType());
    auto convertedElementType = convertedType.getElementType();
    auto oldElementType = op.getMemRefType().getElementType();
    int srcBits = oldElementType.getIntOrFloatBitWidth();
    int dstBits = convertedElementType.getIntOrFloatBitWidth();
    if (dstBits % srcBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "only dstBits % srcBits == 0 supported");
    }

    Location loc = op.getLoc();
    // Special case 0-rank memref loads.
    Value bitsLoad;
    if (convertedType.getRank() == 0) {
      bitsLoad = memref::LoadOp::create(rewriter, loc, adaptor.getMemref(),
                                        ValueRange{});
    } else {
      // Linearize the indices of the original load instruction. Do not account
      // for the scaling yet. This will be accounted for later.
      OpFoldResult linearizedIndices = getLinearizedSrcIndices(
          rewriter, loc, srcBits, adaptor.getIndices(), op.getMemRef());

      Value newLoad = memref::LoadOp::create(
          rewriter, loc, adaptor.getMemref(),
          getIndicesForLoadOrStore(rewriter, loc, linearizedIndices, srcBits,
                                   dstBits));

      // Get the offset and shift the bits to the rightmost.
      // Note, currently only the big-endian is supported.
      Value bitwidthOffset = getOffsetForBitwidth(loc, linearizedIndices,
                                                  srcBits, dstBits, rewriter);
      bitsLoad = arith::ShRSIOp::create(rewriter, loc, newLoad, bitwidthOffset);
    }

    // Get the corresponding bits. If the arith computation bitwidth equals
    // to the emulated bitwidth, we apply a mask to extract the low bits.
    // It is not clear if this case actually happens in practice, but we keep
    // the operations just in case. Otherwise, if the arith computation bitwidth
    // is different from the emulated bitwidth we truncate the result.
    Value result;
    auto resultTy = getTypeConverter()->convertType(oldElementType);
    auto conversionTy =
        resultTy.isInteger()
            ? resultTy
            : IntegerType::get(rewriter.getContext(),
                               resultTy.getIntOrFloatBitWidth());
    if (conversionTy == convertedElementType) {
      auto mask = arith::ConstantOp::create(
          rewriter, loc, convertedElementType,
          rewriter.getIntegerAttr(convertedElementType, (1 << srcBits) - 1));

      result = arith::AndIOp::create(rewriter, loc, bitsLoad, mask);
    } else {
      result = arith::TruncIOp::create(rewriter, loc, conversionTy, bitsLoad);
    }

    if (conversionTy != resultTy) {
      result = arith::BitcastOp::create(rewriter, loc, resultTy, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMemRefMemorySpaceCast
//===----------------------------------------------------------------------===//

struct ConvertMemRefMemorySpaceCast final
    : OpConversionPattern<memref::MemorySpaceCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::MemorySpaceCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getDest().getType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to convert memref type: {0}",
                                      op.getDest().getType()));
    }

    rewriter.replaceOpWithNewOp<memref::MemorySpaceCastOp>(op, newTy,
                                                           adaptor.getSource());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMemRefReinterpretCast
//===----------------------------------------------------------------------===//

/// Output types should be at most one dimensional, so only the 0 or 1
/// dimensional cases are supported.
struct ConvertMemRefReinterpretCast final
    : OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType newTy =
        getTypeConverter()->convertType<MemRefType>(op.getType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to convert memref type: {0}", op.getType()));
    }

    // Only support for 0 or 1 dimensional cases.
    if (op.getType().getRank() > 1) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), "subview with rank > 1 is not supported");
    }

    return convertCastingOp(rewriter, adaptor, op, newTy);
  }
};

//===----------------------------------------------------------------------===//
// ConvertMemrefStore
//===----------------------------------------------------------------------===//

struct ConvertMemrefStore final : OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto convertedType = cast<MemRefType>(adaptor.getMemref().getType());
    int srcBits = op.getMemRefType().getElementTypeBitWidth();
    int dstBits = convertedType.getElementTypeBitWidth();
    auto dstIntegerType = rewriter.getIntegerType(dstBits);
    if (dstBits % srcBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "only dstBits % srcBits == 0 supported");
    }

    Location loc = op.getLoc();

    // Pad the input value with 0s on the left.
    Value input = adaptor.getValue();
    if (!input.getType().isInteger()) {
      input = arith::BitcastOp::create(
          rewriter, loc,
          IntegerType::get(rewriter.getContext(),
                           input.getType().getIntOrFloatBitWidth()),
          input);
    }
    Value extendedInput =
        arith::ExtUIOp::create(rewriter, loc, dstIntegerType, input);

    // Special case 0-rank memref stores. No need for masking.
    if (convertedType.getRank() == 0) {
      memref::AtomicRMWOp::create(rewriter, loc, arith::AtomicRMWKind::assign,
                                  extendedInput, adaptor.getMemref(),
                                  ValueRange{});
      rewriter.eraseOp(op);
      return success();
    }

    OpFoldResult linearizedIndices = getLinearizedSrcIndices(
        rewriter, loc, srcBits, adaptor.getIndices(), op.getMemRef());
    Value storeIndices = getIndicesForLoadOrStore(
        rewriter, loc, linearizedIndices, srcBits, dstBits);
    Value bitwidthOffset = getOffsetForBitwidth(loc, linearizedIndices, srcBits,
                                                dstBits, rewriter);
    Value writeMask = getSubByteWriteMask(loc, linearizedIndices, srcBits,
                                          dstBits, bitwidthOffset, rewriter);
    // Align the value to write with the destination bits
    Value alignedVal =
        arith::ShLIOp::create(rewriter, loc, extendedInput, bitwidthOffset);

    // Clear destination bits
    memref::AtomicRMWOp::create(rewriter, loc, arith::AtomicRMWKind::andi,
                                writeMask, adaptor.getMemref(), storeIndices);
    // Write srcs bits to destination
    memref::AtomicRMWOp::create(rewriter, loc, arith::AtomicRMWKind::ori,
                                alignedVal, adaptor.getMemref(), storeIndices);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMemRefSubview
//===----------------------------------------------------------------------===//

/// Emulating narrow ints on subview have limited support, supporting only
/// static offset and size and stride of 1. Ideally, the subview should be
/// folded away before running narrow type emulation, and this pattern should
/// only run for cases that can't be folded.
struct ConvertMemRefSubview final : OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::SubViewOp subViewOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType newTy =
        getTypeConverter()->convertType<MemRefType>(subViewOp.getType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(
          subViewOp->getLoc(),
          llvm::formatv("failed to convert memref type: {0}",
                        subViewOp.getType()));
    }

    Location loc = subViewOp.getLoc();
    Type convertedElementType = newTy.getElementType();
    Type oldElementType = subViewOp.getType().getElementType();
    int srcBits = oldElementType.getIntOrFloatBitWidth();
    int dstBits = convertedElementType.getIntOrFloatBitWidth();
    if (dstBits % srcBits != 0)
      return rewriter.notifyMatchFailure(
          subViewOp, "only dstBits % srcBits == 0 supported");

    // Only support stride of 1.
    if (llvm::any_of(subViewOp.getStaticStrides(),
                     [](int64_t stride) { return stride != 1; })) {
      return rewriter.notifyMatchFailure(subViewOp->getLoc(),
                                         "stride != 1 is not supported");
    }

    if (!memref::isStaticShapeAndContiguousRowMajor(subViewOp.getType())) {
      return rewriter.notifyMatchFailure(
          subViewOp, "the result memref type is not contiguous");
    }

    auto sizes = subViewOp.getStaticSizes();
    int64_t lastOffset = subViewOp.getStaticOffsets().back();
    // Only support static sizes and offsets.
    if (llvm::is_contained(sizes, ShapedType::kDynamic) ||
        lastOffset == ShapedType::kDynamic) {
      return rewriter.notifyMatchFailure(
          subViewOp->getLoc(), "dynamic size or offset is not supported");
    }

    // Transform the offsets, sizes and strides according to the emulation.
    auto stridedMetadata = memref::ExtractStridedMetadataOp::create(
        rewriter, loc, subViewOp.getViewSource());

    OpFoldResult linearizedIndices;
    auto strides = stridedMetadata.getConstifiedMixedStrides();
    memref::LinearizedMemRefInfo linearizedInfo;
    std::tie(linearizedInfo, linearizedIndices) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, srcBits, dstBits,
            stridedMetadata.getConstifiedMixedOffset(),
            subViewOp.getMixedSizes(), strides,
            getMixedValues(adaptor.getStaticOffsets(), adaptor.getOffsets(),
                           rewriter));

    rewriter.replaceOpWithNewOp<memref::SubViewOp>(
        subViewOp, newTy, adaptor.getSource(), linearizedIndices,
        linearizedInfo.linearizedSize, strides.back());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertMemRefCollapseShape
//===----------------------------------------------------------------------===//

/// Emulating a `memref.collapse_shape` becomes a no-op after emulation given
/// that we flatten memrefs to a single dimension as part of the emulation and
/// there is no dimension to collapse any further.
struct ConvertMemRefCollapseShape final
    : OpConversionPattern<memref::CollapseShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CollapseShapeOp collapseShapeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value srcVal = adaptor.getSrc();
    auto newTy = dyn_cast<MemRefType>(srcVal.getType());
    if (!newTy)
      return failure();

    if (newTy.getRank() != 1)
      return failure();

    rewriter.replaceOp(collapseShapeOp, srcVal);
    return success();
  }
};

/// Emulating a `memref.expand_shape` becomes a no-op after emulation given
/// that we flatten memrefs to a single dimension as part of the emulation and
/// the expansion would just have been undone.
struct ConvertMemRefExpandShape final
    : OpConversionPattern<memref::ExpandShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ExpandShapeOp expandShapeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value srcVal = adaptor.getSrc();
    auto newTy = dyn_cast<MemRefType>(srcVal.getType());
    if (!newTy)
      return failure();

    if (newTy.getRank() != 1)
      return failure();

    rewriter.replaceOp(expandShapeOp, srcVal);
    return success();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Public Interface Definition
//===----------------------------------------------------------------------===//

void memref::populateMemRefNarrowTypeEmulationPatterns(
    const arith::NarrowTypeEmulationConverter &typeConverter,
    RewritePatternSet &patterns) {

  // Populate `memref.*` conversion patterns.
  patterns.add<ConvertMemRefAllocation<memref::AllocOp>,
               ConvertMemRefAllocation<memref::AllocaOp>, ConvertMemRefCopy,
               ConvertMemRefDealloc, ConvertMemRefCollapseShape,
               ConvertMemRefExpandShape, ConvertMemRefLoad, ConvertMemrefStore,
               ConvertMemRefAssumeAlignment, ConvertMemRefMemorySpaceCast,
               ConvertMemRefSubview, ConvertMemRefReinterpretCast>(
      typeConverter, patterns.getContext());
  memref::populateResolveExtractStridedMetadataPatterns(patterns);
}

static SmallVector<int64_t> getLinearizedShape(MemRefType ty, int srcBits,
                                               int dstBits) {
  if (ty.getRank() == 0)
    return {};

  int64_t linearizedShape = 1;
  for (auto shape : ty.getShape()) {
    if (shape == ShapedType::kDynamic)
      return {ShapedType::kDynamic};
    linearizedShape *= shape;
  }
  int scale = dstBits / srcBits;
  // Scale the size to the ceilDiv(linearizedShape, scale)
  // to accomodate all the values.
  linearizedShape = (linearizedShape + scale - 1) / scale;
  return {linearizedShape};
}

void memref::populateMemRefNarrowTypeEmulationConversions(
    arith::NarrowTypeEmulationConverter &typeConverter) {
  typeConverter.addConversion(
      [&typeConverter](MemRefType ty) -> std::optional<Type> {
        Type elementType = ty.getElementType();
        if (!elementType.isIntOrFloat())
          return ty;

        unsigned width = elementType.getIntOrFloatBitWidth();
        unsigned loadStoreWidth = typeConverter.getLoadStoreBitwidth();
        if (width >= loadStoreWidth)
          return ty;

        // Currently only handle innermost stride being 1, checking
        SmallVector<int64_t> strides;
        int64_t offset;
        if (failed(ty.getStridesAndOffset(strides, offset)))
          return nullptr;
        if (!strides.empty() && strides.back() != 1)
          return nullptr;

        auto newElemTy = IntegerType::get(
            ty.getContext(), loadStoreWidth,
            elementType.isInteger()
                ? cast<IntegerType>(elementType).getSignedness()
                : IntegerType::SignednessSemantics::Signless);
        if (!newElemTy)
          return nullptr;

        StridedLayoutAttr layoutAttr;
        // If the offset is 0, we do not need a strided layout as the stride is
        // 1, so we only use the strided layout if the offset is not 0.
        if (offset != 0) {
          if (offset == ShapedType::kDynamic) {
            layoutAttr = StridedLayoutAttr::get(ty.getContext(), offset,
                                                ArrayRef<int64_t>{1});
          } else {
            // Check if the number of bytes are a multiple of the loadStoreWidth
            // and if so, divide it by the loadStoreWidth to get the offset.
            if ((offset * width) % loadStoreWidth != 0)
              return std::nullopt;
            offset = (offset * width) / loadStoreWidth;

            layoutAttr = StridedLayoutAttr::get(ty.getContext(), offset,
                                                ArrayRef<int64_t>{1});
          }
        }

        return MemRefType::get(getLinearizedShape(ty, width, loadStoreWidth),
                               newElemTy, layoutAttr, ty.getMemorySpace());
      });
}
