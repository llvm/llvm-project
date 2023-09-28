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
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

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
  OpFoldResult offsetVal = affine::makeComposedFoldedAffineApply(
      builder, loc, (s0 % scaleFactor) * sourceBits, {srcIdx});
  Value bitOffset = getValueOrCreateConstantIndexOp(builder, loc, offsetVal);
  IntegerType dstType = builder.getIntegerType(targetBits);
  return builder.create<arith::IndexCastOp>(loc, dstType, bitOffset);
}

namespace {

//===----------------------------------------------------------------------===//
// ConvertMemRefAlloc
//===----------------------------------------------------------------------===//

struct ConvertMemRefAlloc final : OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto currentType = op.getMemref().getType().cast<MemRefType>();
    auto newResultType =
        getTypeConverter()->convertType(op.getType()).dyn_cast<MemRefType>();
    if (!newResultType) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to convert memref type: {0}", op.getType()));
    }

    // Special case zero-rank memrefs.
    if (currentType.getRank() == 0) {
      rewriter.replaceOpWithNewOp<memref::AllocOp>(
          op, newResultType, ValueRange{}, adaptor.getSymbolOperands(),
          adaptor.getAlignmentAttr());
      return success();
    }

    Location loc = op.getLoc();
    OpFoldResult zero = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> indices(currentType.getRank(), zero);

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

    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op, newResultType, dynamicLinearizedSize, adaptor.getSymbolOperands(),
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
        op, adaptor.getMemref(), adaptor.getAlignmentAttr());
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
    auto convertedType = adaptor.getMemref().getType().cast<MemRefType>();
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
      bitsLoad = rewriter.create<memref::LoadOp>(loc, adaptor.getMemref(),
                                                 ValueRange{});
    } else {
      SmallVector<OpFoldResult> indices =
          getAsOpFoldResult(adaptor.getIndices());

      auto stridedMetadata = rewriter.create<memref::ExtractStridedMetadataOp>(
          loc, op.getMemRef());

      // Linearize the indices of the original load instruction. Do not account
      // for the scaling yet. This will be accounted for later.
      OpFoldResult linearizedIndices;
      std::tie(std::ignore, linearizedIndices) =
          memref::getLinearizedMemRefOffsetAndSize(
              rewriter, loc, srcBits, srcBits,
              stridedMetadata.getConstifiedMixedOffset(),
              stridedMetadata.getConstifiedMixedSizes(),
              stridedMetadata.getConstifiedMixedStrides(), indices);

      AffineExpr s0;
      bindSymbols(rewriter.getContext(), s0);
      int64_t scaler = dstBits / srcBits;
      OpFoldResult scaledLinearizedIndices =
          affine::makeComposedFoldedAffineApply(
              rewriter, loc, s0.floorDiv(scaler), {linearizedIndices});
      Value newLoad = rewriter.create<memref::LoadOp>(
          loc, adaptor.getMemref(),
          getValueOrCreateConstantIndexOp(rewriter, loc,
                                          scaledLinearizedIndices));

      // Get the offset and shift the bits to the rightmost.
      // Note, currently only the big-endian is supported.
      Value bitwidthOffset = getOffsetForBitwidth(loc, linearizedIndices,
                                                  srcBits, dstBits, rewriter);
      bitsLoad = rewriter.create<arith::ShRSIOp>(loc, newLoad, bitwidthOffset);
    }

    // Get the corresponding bits. If the arith computation bitwidth equals
    // to the emulated bitwidth, we apply a mask to extract the low bits.
    // It is not clear if this case actually happens in practice, but we keep
    // the operations just in case. Otherwise, if the arith computation bitwidth
    // is different from the emulated bitwidth we truncate the result.
    Operation *result;
    auto resultTy = getTypeConverter()->convertType(oldElementType);
    if (resultTy == convertedElementType) {
      auto mask = rewriter.create<arith::ConstantOp>(
          loc, convertedElementType,
          rewriter.getIntegerAttr(convertedElementType, (1 << srcBits) - 1));

      result = rewriter.create<arith::AndIOp>(loc, bitsLoad, mask);
    } else {
      result = rewriter.create<arith::TruncIOp>(loc, resultTy, bitsLoad);
    }

    rewriter.replaceOp(op, result->getResult(0));
    return success();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Public Interface Definition
//===----------------------------------------------------------------------===//

void memref::populateMemRefNarrowTypeEmulationPatterns(
    arith::NarrowTypeEmulationConverter &typeConverter,
    RewritePatternSet &patterns) {

  // Populate `memref.*` conversion patterns.
  patterns
      .add<ConvertMemRefAlloc, ConvertMemRefLoad, ConvertMemRefAssumeAlignment>(
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
        auto intTy = dyn_cast<IntegerType>(ty.getElementType());
        if (!intTy)
          return ty;

        unsigned width = intTy.getWidth();
        unsigned loadStoreWidth = typeConverter.getLoadStoreBitwidth();
        if (width >= loadStoreWidth)
          return ty;

        // Currently only handle innermost stride being 1, checking
        SmallVector<int64_t> strides;
        int64_t offset;
        if (failed(getStridesAndOffset(ty, strides, offset)))
          return std::nullopt;
        if (!strides.empty() && strides.back() != 1)
          return std::nullopt;

        auto newElemTy = IntegerType::get(ty.getContext(), loadStoreWidth,
                                          intTy.getSignedness());
        if (!newElemTy)
          return std::nullopt;

        StridedLayoutAttr layoutAttr;
        if (offset != 0) {
          // Check if the number of bytes are a multiple of the loadStoreWidth
          // and if so, divide it by the loadStoreWidth to get the offset.
          if ((offset * width) % loadStoreWidth != 0)
            return std::nullopt;
          offset = (offset * width) / loadStoreWidth;

          layoutAttr = StridedLayoutAttr::get(ty.getContext(), offset,
                                              ArrayRef<int64_t>{1});
        }

        return MemRefType::get(getLinearizedShape(ty, width, loadStoreWidth),
                               newElemTy, layoutAttr, ty.getMemorySpace());
      });
}
