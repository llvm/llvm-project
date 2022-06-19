//===- ControlFlowToSPIRV.cpp - ControlFlow to SPIR-V Patterns ------------===//
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

#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h"
#include "../SPIRVCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/SPIRV/Utils/LayoutUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cf-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

namespace {

/// Converts cf.br to spv.Branch.
struct BranchOpPattern final : public OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern<cf::BranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<spirv::BranchOp>(op, op.getDest(),
                                                 adaptor.getDestOperands());
    return success();
  }
};

/// Converts cf.cond_br to spv.BranchConditional.
struct CondBranchOpPattern final
    : public OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern<cf::CondBranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<spirv::BranchConditionalOp>(
        op, op.getCondition(), op.getTrueDest(), adaptor.getTrueDestOperands(),
        op.getFalseDest(), adaptor.getFalseDestOperands());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::cf::populateControlFlowToSPIRVPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  patterns.add<BranchOpPattern, CondBranchOpPattern>(typeConverter, context);
}
