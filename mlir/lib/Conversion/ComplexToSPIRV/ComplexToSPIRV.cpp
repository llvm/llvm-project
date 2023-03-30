//===- ComplexToSPIRV.cpp - Complex to SPIR-V Patterns --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Complex dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ComplexToSPIRV/ComplexToSPIRV.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "complex-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

namespace {

struct CreateOpPattern final : OpConversionPattern<complex::CreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::CreateOp createOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(createOp.getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(createOp,
                                         "unable to convert result type");

    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(
        createOp, spirvType, adaptor.getOperands());
    return success();
  }
};

struct ReOpPattern final : OpConversionPattern<complex::ReOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::ReOp reOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(reOp.getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(reOp, "unable to convert result type");

    rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
        reOp, adaptor.getComplex(), llvm::ArrayRef(0));
    return success();
  }
};

struct ImOpPattern final : OpConversionPattern<complex::ImOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(complex::ImOp imOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type spirvType = getTypeConverter()->convertType(imOp.getType());
    if (!spirvType)
      return rewriter.notifyMatchFailure(imOp, "unable to convert result type");

    rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
        imOp, adaptor.getComplex(), llvm::ArrayRef(1));
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateComplexToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  patterns.add<CreateOpPattern, ReOpPattern, ImOpPattern>(typeConverter,
                                                          context);
}
