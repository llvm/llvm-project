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
static Value getOffsetForBitwidth(Location loc, Value srcIdx, int sourceBits,
                                  int targetBits, OpBuilder &builder) {
  assert(targetBits % sourceBits == 0);
  IntegerType targetType = builder.getIntegerType(targetBits);
  IntegerAttr idxAttr =
      builder.getIntegerAttr(targetType, targetBits / sourceBits);
  auto idx = builder.create<arith::ConstantOp>(loc, targetType, idxAttr);
  IntegerAttr srcBitsAttr = builder.getIntegerAttr(targetType, sourceBits);
  auto srcBitsValue =
      builder.create<arith::ConstantOp>(loc, targetType, srcBitsAttr);
  auto m = builder.create<arith::RemUIOp>(loc, srcIdx, idx);
  return builder.create<arith::MulIOp>(loc, targetType, m, srcBitsValue);
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
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to convert memref type: {0}", op.getType()));
    }

    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op, newTy, adaptor.getDynamicSizes(), adaptor.getSymbolOperands(),
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
    Type newTy = getTypeConverter()->convertType(op.getMemRefType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to convert memref type: {0}",
                                      op.getMemRefType()));
    }

    if (op.getMemRefType() == newTy)
      return failure();

    auto loc = op.getLoc();
    auto sourceType = cast<MemRefType>(adaptor.getMemref().getType());
    unsigned sourceRank = sourceType.getRank();
    SmallVector<Value> indices = adaptor.getIndices();
    assert(indices.size() == sourceRank);

    auto srcElementType = sourceType.getElementType();
    auto oldElementType = op.getMemRefType().getElementType();
    int srcBits = oldElementType.getIntOrFloatBitWidth();
    int dstBits = srcElementType.getIntOrFloatBitWidth();
    if (dstBits % srcBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "only dstBits % srcBits == 0 supported");
    }

    auto stridedMetadata = rewriter.create<memref::ExtractStridedMetadataOp>(
        loc, adaptor.getMemref());

    Value newLoad, lastIdx;
    if (sourceRank == 0) {
      newLoad = rewriter.create<memref::LoadOp>(
          loc, srcElementType, adaptor.getMemref(), adaptor.getIndices());

      lastIdx = stridedMetadata.getOffset();
    } else {
      auto [reinterpret, linearizedOffset] =
          memref::getLinearizeMemRefAndOffset(loc, sourceType, srcBits, dstBits,
                                              adaptor.getIndices(),
                                              stridedMetadata, rewriter);

      newLoad = rewriter.create<memref::LoadOp>(loc, srcElementType,
                                                reinterpret, linearizedOffset);

      lastIdx = adaptor.getIndices().back();
    }

    // Get the offset and shift the bits to the rightmost.
    // Note, currently only the big-endian is supported.
    auto castLastIdx =
        rewriter.create<arith::IndexCastUIOp>(loc, srcElementType, lastIdx);

    Value BitwidthOffset =
        getOffsetForBitwidth(loc, castLastIdx, srcBits, dstBits, rewriter);
    auto bitsLoad =
        rewriter.create<arith::ShRSIOp>(loc, newLoad, BitwidthOffset);

    // Get the corresponding bits. If the arith computation bitwidth equals
    // to the emulated bitwidth, we apply a mask to extract the low bits.
    // It is not clear if this case actually happens in practice, but we keep
    // the operations just in case. Otherwise, if the arith computation bitwidth
    // is different from the emulated bitwidth we truncate the result.
    Operation *result;
    auto resultTy = getTypeConverter()->convertType(oldElementType);
    if (resultTy == srcElementType) {
      auto mask = rewriter.create<arith::ConstantOp>(
          loc, srcElementType,
          rewriter.getIntegerAttr(srcElementType, (1 << srcBits) - 1));

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

        auto newElemTy = IntegerType::get(ty.getContext(), loadStoreWidth,
                                          intTy.getSignedness());
        if (!newElemTy)
          return std::nullopt;

        return ty.cloneWith(std::nullopt, newElemTy);
      });
}
