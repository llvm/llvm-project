//===- FuncToSPIRV.cpp - Func to SPIR-V Patterns ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Func dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "../SPIRVCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/SPIRV/Utils/LayoutUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "func-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

// Note that DRR cannot be used for the patterns in this file: we may need to
// convert type along the way, which requires ConversionPattern. DRR generates
// normal RewritePattern.

namespace {

/// Converts func.return to spv.Return.
class ReturnOpPattern final : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (returnOp.getNumOperands() > 1)
      return failure();

    if (returnOp.getNumOperands() == 1) {
      rewriter.replaceOpWithNewOp<spirv::ReturnValueOp>(
          returnOp, adaptor.getOperands()[0]);
    } else {
      rewriter.replaceOpWithNewOp<spirv::ReturnOp>(returnOp);
    }
    return success();
  }
};

/// Converts func.call to spv.FunctionCall.
class CallOpPattern final : public OpConversionPattern<func::CallOp> {
public:
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // multiple results func was not converted to spv.func
    if (callOp.getNumResults() > 1)
      return failure();
    if (callOp.getNumResults() == 1) {
      auto resultType =
          getTypeConverter()->convertType(callOp.getResult(0).getType());
      if (!resultType)
        return failure();
      rewriter.replaceOpWithNewOp<spirv::FunctionCallOp>(
          callOp, resultType, adaptor.getOperands(), callOp->getAttrs());
    } else {
      rewriter.replaceOpWithNewOp<spirv::FunctionCallOp>(
          callOp, TypeRange(), adaptor.getOperands(), callOp->getAttrs());
    }
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateFuncToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  patterns.add<ReturnOpPattern, CallOpPattern>(typeConverter, context);
}
