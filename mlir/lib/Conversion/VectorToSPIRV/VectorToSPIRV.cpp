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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
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

    if (dstType == adaptor.getSource().getType())
      rewriter.replaceOp(bitcastOp, adaptor.getSource());
    else
      rewriter.replaceOpWithNewOp<spirv::BitcastOp>(bitcastOp, dstType,
                                                    adaptor.getSource());

    return success();
  }
};

struct VectorBroadcastConvert final
    : public OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp broadcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (broadcastOp.getSource().getType().isa<VectorType>() ||
        !spirv::CompositeType::isValid(broadcastOp.getVectorType()))
      return failure();
    SmallVector<Value, 4> source(broadcastOp.getVectorType().getNumElements(),
                                 adaptor.getSource());
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

    if (adaptor.getVector().getType().isa<spirv::ScalarType>()) {
      rewriter.replaceOp(extractOp, adaptor.getVector());
      return success();
    }

    int32_t id = getFirstIntValue(extractOp.getPosition());
    rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
        extractOp, adaptor.getVector(), id);
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

    uint64_t offset = getFirstIntValue(extractOp.getOffsets());
    uint64_t size = getFirstIntValue(extractOp.getSizes());
    uint64_t stride = getFirstIntValue(extractOp.getStrides());
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
        fmaOp, fmaOp.getType(), adaptor.getLhs(), adaptor.getRhs(),
        adaptor.getAcc());
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
      rewriter.replaceOp(insertOp, adaptor.getSource());
      return success();
    }

    if (insertOp.getSourceType().isa<VectorType>() ||
        !spirv::CompositeType::isValid(insertOp.getDestVectorType()))
      return failure();
    int32_t id = getFirstIntValue(insertOp.getPosition());
    rewriter.replaceOpWithNewOp<spirv::CompositeInsertOp>(
        insertOp, adaptor.getSource(), adaptor.getDest(), id);
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
        extractElementOp, extractElementOp.getType(), adaptor.getVector(),
        extractElementOp.getPosition());
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
        insertElementOp, insertElementOp.getType(), insertElementOp.getDest(),
        adaptor.getSource(), insertElementOp.getPosition());
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

    uint64_t stride = getFirstIntValue(insertOp.getStrides());
    if (stride != 1)
      return failure();
    uint64_t offset = getFirstIntValue(insertOp.getOffsets());

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

class VectorSplatPattern final : public OpConversionPattern<vector::SplatOp> {
public:
  using OpConversionPattern<vector::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType dstVecType = op.getType();
    if (!spirv::CompositeType::isValid(dstVecType))
      return failure();
    SmallVector<Value, 4> source(dstVecType.getNumElements(),
                                 adaptor.getInput());
    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, dstVecType,
                                                             source);
    return success();
  }
};

struct VectorShuffleOpConvert final
    : public OpConversionPattern<vector::ShuffleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ShuffleOp shuffleOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto oldResultType = shuffleOp.getVectorType();
    if (!spirv::CompositeType::isValid(oldResultType))
      return failure();
    auto newResultType = getTypeConverter()->convertType(oldResultType);

    auto oldSourceType = shuffleOp.getV1VectorType();
    if (oldSourceType.getNumElements() > 1) {
      SmallVector<int32_t, 4> components = llvm::to_vector<4>(
          llvm::map_range(shuffleOp.getMask(), [](Attribute attr) -> int32_t {
            return attr.cast<IntegerAttr>().getValue().getZExtValue();
          }));
      rewriter.replaceOpWithNewOp<spirv::VectorShuffleOp>(
          shuffleOp, newResultType, adaptor.getV1(), adaptor.getV2(),
          rewriter.getI32ArrayAttr(components));
      return success();
    }

    SmallVector<Value, 2> oldOperands = {adaptor.getV1(), adaptor.getV2()};
    SmallVector<Value, 4> newOperands;
    newOperands.reserve(oldResultType.getNumElements());
    for (const APInt &i : shuffleOp.getMask().getAsValueRange<IntegerAttr>()) {
      newOperands.push_back(oldOperands[i.getZExtValue()]);
    }
    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        shuffleOp, newResultType, newOperands);

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
               VectorInsertStridedSliceOpConvert, VectorShuffleOpConvert,
               VectorSplatPattern>(typeConverter, patterns.getContext());
}
