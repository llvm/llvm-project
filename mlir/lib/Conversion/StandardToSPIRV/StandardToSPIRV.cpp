//===- StandardToSPIRV.cpp - Standard to SPIR-V Patterns ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert standard dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "../SPIRVCommon/Pattern.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/SPIRV/Utils/LayoutUtils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "std-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

// Note that DRR cannot be used for the patterns in this file: we may need to
// convert type along the way, which requires ConversionPattern. DRR generates
// normal RewritePattern.

namespace {

/// Converts std.return to spv.Return.
class ReturnOpPattern final : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.select to spv.Select.
class SelectOpPattern final : public OpConversionPattern<SelectOp> {
public:
  using OpConversionPattern<SelectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.splat to spv.CompositeConstruct.
class SplatPattern final : public OpConversionPattern<SplatOp> {
public:
  using OpConversionPattern<SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts tensor.extract into loading using access chains from SPIR-V local
/// variables.
class TensorExtractPattern final
    : public OpConversionPattern<tensor::ExtractOp> {
public:
  TensorExtractPattern(TypeConverter &typeConverter, MLIRContext *context,
                       int64_t threshold, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        byteCountThreshold(threshold) {}

  LogicalResult
  matchAndRewrite(tensor::ExtractOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TensorType tensorType = extractOp.tensor().getType().cast<TensorType>();

    if (!tensorType.hasStaticShape())
      return rewriter.notifyMatchFailure(extractOp, "non-static tensor");

    if (tensorType.getNumElements() * tensorType.getElementTypeBitWidth() >
        byteCountThreshold * 8)
      return rewriter.notifyMatchFailure(extractOp,
                                         "exceeding byte count threshold");

    Location loc = extractOp.getLoc();

    int64_t rank = tensorType.getRank();
    SmallVector<int64_t, 4> strides(rank, 1);
    for (int i = rank - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * tensorType.getDimSize(i + 1);
    }

    Type varType = spirv::PointerType::get(adaptor.tensor().getType(),
                                           spirv::StorageClass::Function);

    spirv::VariableOp varOp;
    if (adaptor.tensor().getDefiningOp<spirv::ConstantOp>()) {
      varOp = rewriter.create<spirv::VariableOp>(
          loc, varType, spirv::StorageClass::Function,
          /*initializer=*/adaptor.tensor());
    } else {
      // Need to store the value to the local variable. It's questionable
      // whether we want to support such case though.
      return failure();
    }

    auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
    auto indexType = typeConverter.getIndexType();

    Value index = spirv::linearizeIndex(adaptor.indices(), strides,
                                        /*offset=*/0, indexType, loc, rewriter);
    auto acOp = rewriter.create<spirv::AccessChainOp>(loc, varOp, index);

    rewriter.replaceOpWithNewOp<spirv::LoadOp>(extractOp, acOp);

    return success();
  }

private:
  int64_t byteCountThreshold;
};

} // namespace

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult
ReturnOpPattern::matchAndRewrite(ReturnOp returnOp, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  if (returnOp.getNumOperands() > 1)
    return failure();

  if (returnOp.getNumOperands() == 1) {
    rewriter.replaceOpWithNewOp<spirv::ReturnValueOp>(returnOp,
                                                      adaptor.getOperands()[0]);
  } else {
    rewriter.replaceOpWithNewOp<spirv::ReturnOp>(returnOp);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult
SelectOpPattern::matchAndRewrite(SelectOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, adaptor.getCondition(),
                                               adaptor.getTrueValue(),
                                               adaptor.getFalseValue());
  return success();
}

//===----------------------------------------------------------------------===//
// SplatOp
//===----------------------------------------------------------------------===//

LogicalResult
SplatPattern::matchAndRewrite(SplatOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  auto dstVecType = op.getType().dyn_cast<VectorType>();
  if (!dstVecType || !spirv::CompositeType::isValid(dstVecType))
    return failure();
  SmallVector<Value, 4> source(dstVecType.getNumElements(), adaptor.getInput());
  rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, dstVecType,
                                                           source);
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

namespace mlir {
void populateStandardToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  patterns.add<
      // Unary and binary patterns
      spirv::UnaryAndBinaryOpPattern<MaxFOp, spirv::GLSLFMaxOp>,
      spirv::UnaryAndBinaryOpPattern<MaxSIOp, spirv::GLSLSMaxOp>,
      spirv::UnaryAndBinaryOpPattern<MaxUIOp, spirv::GLSLUMaxOp>,
      spirv::UnaryAndBinaryOpPattern<MinFOp, spirv::GLSLFMinOp>,
      spirv::UnaryAndBinaryOpPattern<MinSIOp, spirv::GLSLSMinOp>,
      spirv::UnaryAndBinaryOpPattern<MinUIOp, spirv::GLSLUMinOp>,

      ReturnOpPattern, SelectOpPattern, SplatPattern>(typeConverter, context);
}

void populateTensorToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                   int64_t byteCountThreshold,
                                   RewritePatternSet &patterns) {
  patterns.add<TensorExtractPattern>(typeConverter, patterns.getContext(),
                                     byteCountThreshold);
}

} // namespace mlir
