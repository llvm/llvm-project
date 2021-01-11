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
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct VectorBroadcastConvert final
    : public OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp broadcastOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (broadcastOp.source().getType().isa<VectorType>() ||
        !spirv::CompositeType::isValid(broadcastOp.getVectorType()))
      return failure();
    vector::BroadcastOp::Adaptor adaptor(operands);
    SmallVector<Value, 4> source(broadcastOp.getVectorType().getNumElements(),
                                 adaptor.source());
    Value construct = rewriter.create<spirv::CompositeConstructOp>(
        broadcastOp.getLoc(), broadcastOp.getVectorType(), source);
    rewriter.replaceOp(broadcastOp, construct);
    return success();
  }
};

struct VectorExtractOpConvert final
    : public OpConversionPattern<vector::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractOp extractOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (extractOp.getType().isa<VectorType>() ||
        !spirv::CompositeType::isValid(extractOp.getVectorType()))
      return failure();
    vector::ExtractOp::Adaptor adaptor(operands);
    int32_t id = extractOp.position().begin()->cast<IntegerAttr>().getInt();
    Value newExtract = rewriter.create<spirv::CompositeExtractOp>(
        extractOp.getLoc(), adaptor.vector(), id);
    rewriter.replaceOp(extractOp, newExtract);
    return success();
  }
};

struct VectorInsertOpConvert final
    : public OpConversionPattern<vector::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InsertOp insertOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (insertOp.getSourceType().isa<VectorType>() ||
        !spirv::CompositeType::isValid(insertOp.getDestVectorType()))
      return failure();
    vector::InsertOp::Adaptor adaptor(operands);
    int32_t id = insertOp.position().begin()->cast<IntegerAttr>().getInt();
    Value newInsert = rewriter.create<spirv::CompositeInsertOp>(
        insertOp.getLoc(), adaptor.source(), adaptor.dest(), id);
    rewriter.replaceOp(insertOp, newInsert);
    return success();
  }
};

struct VectorExtractElementOpConvert final
    : public OpConversionPattern<vector::ExtractElementOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractElementOp extractElementOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!spirv::CompositeType::isValid(extractElementOp.getVectorType()))
      return failure();
    vector::ExtractElementOp::Adaptor adaptor(operands);
    Value newExtractElement = rewriter.create<spirv::VectorExtractDynamicOp>(
        extractElementOp.getLoc(), extractElementOp.getType(), adaptor.vector(),
        extractElementOp.position());
    rewriter.replaceOp(extractElementOp, newExtractElement);
    return success();
  }
};

struct VectorInsertElementOpConvert final
    : public OpConversionPattern<vector::InsertElementOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InsertElementOp insertElementOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!spirv::CompositeType::isValid(insertElementOp.getDestVectorType()))
      return failure();
    vector::InsertElementOp::Adaptor adaptor(operands);
    Value newInsertElement = rewriter.create<spirv::VectorInsertDynamicOp>(
        insertElementOp.getLoc(), insertElementOp.getType(),
        insertElementOp.dest(), adaptor.source(), insertElementOp.position());
    rewriter.replaceOp(insertElementOp, newInsertElement);
    return success();
  }
};

} // namespace

void mlir::populateVectorToSPIRVPatterns(MLIRContext *context,
                                         SPIRVTypeConverter &typeConverter,
                                         OwningRewritePatternList &patterns) {
  patterns.insert<VectorBroadcastConvert, VectorExtractOpConvert,
                  VectorInsertOpConvert, VectorExtractElementOpConvert,
                  VectorInsertElementOpConvert>(typeConverter, context);
}
