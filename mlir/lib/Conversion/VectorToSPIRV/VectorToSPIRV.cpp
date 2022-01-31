//===- VectorToSPIRV.cpp - Vector to SPIR-V Patterns ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Vector dialect to SPIRV dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"

#include "../PassDetail.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include <numeric>

using namespace mlir;

/// Gets the first integer value from `attr`, assuming it is an integer array
/// attribute.
static uint64_t getFirstIntValue(ArrayAttr attr) {
  return (*attr.getAsValueRange<IntegerAttr>().begin()).getZExtValue();
}

namespace {

struct VectorBitcastConvert final
    : public OpConversionPattern<vector::BitCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BitCastOp bitcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = getTypeConverter()->convertType(bitcastOp.getType());
    if (!dstType)
      return failure();

    if (dstType == adaptor.source().getType())
      rewriter.replaceOp(bitcastOp, adaptor.source());
    else
      rewriter.replaceOpWithNewOp<spirv::BitcastOp>(bitcastOp, dstType,
                                                    adaptor.source());

    return success();
  }
};

struct VectorBroadcastConvert final
    : public OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp broadcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (broadcastOp.source().getType().isa<VectorType>() ||
        !spirv::CompositeType::isValid(broadcastOp.getVectorType()))
      return failure();
    SmallVector<Value, 4> source(broadcastOp.getVectorType().getNumElements(),
                                 adaptor.source());
    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        broadcastOp, broadcastOp.getVectorType(), source);
    return success();
  }
};

struct VectorExtractOpConvert final
    : public OpConversionPattern<vector::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only support extracting a scalar value now.
    VectorType resultVectorType = extractOp.getType().dyn_cast<VectorType>();
    if (resultVectorType && resultVectorType.getNumElements() > 1)
      return failure();

    auto dstType = getTypeConverter()->convertType(extractOp.getType());
    if (!dstType)
      return failure();

    if (adaptor.vector().getType().isa<spirv::ScalarType>()) {
      rewriter.replaceOp(extractOp, adaptor.vector());
      return success();
    }

    int32_t id = getFirstIntValue(extractOp.position());
    rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
        extractOp, adaptor.vector(), id);
    return success();
  }
};

struct VectorExtractStridedSliceOpConvert final
    : public OpConversionPattern<vector::ExtractStridedSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractStridedSliceOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = getTypeConverter()->convertType(extractOp.getType());
    if (!dstType)
      return failure();


    uint64_t offset = getFirstIntValue(extractOp.offsets());
    uint64_t size = getFirstIntValue(extractOp.sizes());
    uint64_t stride = getFirstIntValue(extractOp.strides());
    if (stride != 1)
      return failure();

    Value srcVector = adaptor.getOperands().front();

    // Extract vector<1xT> case.
    if (dstType.isa<spirv::ScalarType>()) {
      rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(extractOp,
                                                             srcVector, offset);
      return success();
    }

    SmallVector<int32_t, 2> indices(size);
    std::iota(indices.begin(), indices.end(), offset);

    rewriter.replaceOpWithNewOp<spirv::VectorShuffleOp>(
        extractOp, dstType, srcVector, srcVector,
        rewriter.getI32ArrayAttr(indices));

    return success();
  }
};

struct VectorFmaOpConvert final : public OpConversionPattern<vector::FMAOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::FMAOp fmaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!spirv::CompositeType::isValid(fmaOp.getVectorType()))
      return failure();
    rewriter.replaceOpWithNewOp<spirv::GLSLFmaOp>(
        fmaOp, fmaOp.getType(), adaptor.lhs(), adaptor.rhs(), adaptor.acc());
    return success();
  }
};

struct VectorInsertOpConvert final
    : public OpConversionPattern<vector::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InsertOp insertOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Special case for inserting scalar values into size-1 vectors.
    if (insertOp.getSourceType().isIntOrFloat() &&
        insertOp.getDestVectorType().getNumElements() == 1) {
      rewriter.replaceOp(insertOp, adaptor.source());
      return success();
    }

    if (insertOp.getSourceType().isa<VectorType>() ||
        !spirv::CompositeType::isValid(insertOp.getDestVectorType()))
      return failure();
    int32_t id = getFirstIntValue(insertOp.position());
    rewriter.replaceOpWithNewOp<spirv::CompositeInsertOp>(
        insertOp, adaptor.source(), adaptor.dest(), id);
    return success();
  }
};

struct VectorExtractElementOpConvert final
    : public OpConversionPattern<vector::ExtractElementOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractElementOp extractElementOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!spirv::CompositeType::isValid(extractElementOp.getVectorType()))
      return failure();
    rewriter.replaceOpWithNewOp<spirv::VectorExtractDynamicOp>(
        extractElementOp, extractElementOp.getType(), adaptor.vector(),
        extractElementOp.position());
    return success();
  }
};

struct VectorInsertElementOpConvert final
    : public OpConversionPattern<vector::InsertElementOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InsertElementOp insertElementOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!spirv::CompositeType::isValid(insertElementOp.getDestVectorType()))
      return failure();
    rewriter.replaceOpWithNewOp<spirv::VectorInsertDynamicOp>(
        insertElementOp, insertElementOp.getType(), insertElementOp.dest(),
        adaptor.source(), insertElementOp.position());
    return success();
  }
};

struct VectorInsertStridedSliceOpConvert final
    : public OpConversionPattern<vector::InsertStridedSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InsertStridedSliceOp insertOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value srcVector = adaptor.getOperands().front();
    Value dstVector = adaptor.getOperands().back();

    uint64_t stride = getFirstIntValue(insertOp.strides());
    if (stride != 1)
      return failure();
    uint64_t offset = getFirstIntValue(insertOp.offsets());

    if (srcVector.getType().isa<spirv::ScalarType>()) {
      assert(!dstVector.getType().isa<spirv::ScalarType>());
      rewriter.replaceOpWithNewOp<spirv::CompositeInsertOp>(
          insertOp, dstVector.getType(), srcVector, dstVector,
          rewriter.getI32ArrayAttr(offset));
      return success();
    }

    uint64_t totalSize =
        dstVector.getType().cast<VectorType>().getNumElements();
    uint64_t insertSize =
        srcVector.getType().cast<VectorType>().getNumElements();

    SmallVector<int32_t, 2> indices(totalSize);
    std::iota(indices.begin(), indices.end(), 0);
    std::iota(indices.begin() + offset, indices.begin() + offset + insertSize,
              totalSize);

    rewriter.replaceOpWithNewOp<spirv::VectorShuffleOp>(
        insertOp, dstVector.getType(), dstVector, srcVector,
        rewriter.getI32ArrayAttr(indices));

    return success();
  }
};

} // namespace

void mlir::populateVectorToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  patterns.add<VectorBitcastConvert, VectorBroadcastConvert,
               VectorExtractElementOpConvert, VectorExtractOpConvert,
               VectorExtractStridedSliceOpConvert, VectorFmaOpConvert,
               VectorInsertElementOpConvert, VectorInsertOpConvert,
               VectorInsertStridedSliceOpConvert>(typeConverter,
                                                  patterns.getContext());
}
