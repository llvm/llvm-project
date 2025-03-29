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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "cf-to-spirv-pattern"

using namespace mlir;

/// Legailze target block arguments.
static LogicalResult legalizeBlockArguments(Block &block, Operation *op,
                                            PatternRewriter &rewriter,
                                            const TypeConverter &converter) {
  auto builder = OpBuilder::atBlockBegin(&block);
  for (unsigned i = 0; i < block.getNumArguments(); ++i) {
    BlockArgument arg = block.getArgument(i);
    if (converter.isLegal(arg.getType()))
      continue;
    Type ty = arg.getType();
    Type newTy = converter.convertType(ty);
    if (!newTy) {
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("failed to legalize type for argument {0})", arg));
    }
    unsigned argNum = arg.getArgNumber();
    Location loc = arg.getLoc();
    Value newArg = block.insertArgument(argNum, newTy, loc);
    Value convertedValue = converter.materializeSourceConversion(
        builder, op->getLoc(), ty, newArg);
    if (!convertedValue) {
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("failed to cast new argument {0} to type {1})",
                            newArg, ty));
    }
    arg.replaceAllUsesWith(convertedValue);
    block.eraseArgument(argNum + 1);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

namespace {
/// Converts cf.br to spirv.Branch.
struct BranchOpPattern final : OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(legalizeBlockArguments(*op.getDest(), op, rewriter,
                                      *getTypeConverter())))
      return failure();

    rewriter.replaceOpWithNewOp<spirv::BranchOp>(op, op.getDest(),
                                                 adaptor.getDestOperands());
    return success();
  }
};

/// Converts cf.cond_br to spirv.BranchConditional.
struct CondBranchOpPattern final : OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(legalizeBlockArguments(*op.getTrueDest(), op, rewriter,
                                      *getTypeConverter())))
      return failure();

    if (failed(legalizeBlockArguments(*op.getFalseDest(), op, rewriter,
                                      *getTypeConverter())))
      return failure();

    rewriter.replaceOpWithNewOp<spirv::BranchConditionalOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::cf::populateControlFlowToSPIRVPatterns(
    const SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  patterns.add<BranchOpPattern, CondBranchOpPattern>(typeConverter, context);
}
