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
    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        broadcastOp, broadcastOp.getVectorType(), source);
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
    rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
        extractOp, adaptor.vector(), id);
    return success();
  }
};

struct VectorFmaOpConvert final : public OpConversionPattern<vector::FMAOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::FMAOp fmaOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!spirv::CompositeType::isValid(fmaOp.getVectorType()))
      return failure();
    vector::FMAOp::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<spirv::GLSLFmaOp>(
        fmaOp, fmaOp.getType(), adaptor.lhs(), adaptor.rhs(), adaptor.acc());
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
    rewriter.replaceOpWithNewOp<spirv::CompositeInsertOp>(
        insertOp, adaptor.source(), adaptor.dest(), id);
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
  matchAndRewrite(vector::InsertElementOp insertElementOp,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!spirv::CompositeType::isValid(insertElementOp.getDestVectorType()))
      return failure();
    vector::InsertElementOp::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<spirv::VectorInsertDynamicOp>(
        insertElementOp, insertElementOp.getType(), insertElementOp.dest(),
        adaptor.source(), insertElementOp.position());
    return success();
  }
};

} // namespace

void mlir::populateVectorToSPIRVPatterns(MLIRContext *context,
                                         SPIRVTypeConverter &typeConverter,
                                         OwningRewritePatternList &patterns) {
  patterns.insert<VectorBroadcastConvert, VectorExtractElementOpConvert,
                  VectorExtractOpConvert, VectorFmaOpConvert,
                  VectorInsertOpConvert, VectorInsertElementOpConvert>(
      typeConverter, context);
}
