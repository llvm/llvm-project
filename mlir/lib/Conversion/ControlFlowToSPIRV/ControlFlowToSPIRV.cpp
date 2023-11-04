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
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "cf-to-spirv-pattern"

using namespace mlir;

/// Legailze target block arguments.
static void legalizeBlockArguments(Block &block,
                                   const TypeConverter &converter) {
  auto builder = OpBuilder::atBlockBegin(&block);
  for (unsigned i = 0; i < block.getNumArguments(); ++i) {
    const auto arg = block.getArgument(i);
    if (converter.isLegal(arg.getType()))
      continue;
    unsigned argNum = arg.getArgNumber();
    Location loc = arg.getLoc();
    Type ty = arg.getType();
    Type newTy = converter.convertType(ty);
    Value newArg = block.insertArgument(argNum, newTy, loc);
    auto cast = builder.create<UnrealizedConversionCastOp>(loc, ty, newArg);
    arg.replaceAllUsesWith(cast.getResult(0));
    block.eraseArgument(argNum + 1);
  }
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
    legalizeBlockArguments(*op.getDest(), *getTypeConverter());

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
    legalizeBlockArguments(*op.getTrueDest(), *getTypeConverter());
    legalizeBlockArguments(*op.getFalseDest(), *getTypeConverter());

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
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  patterns.add<BranchOpPattern, CondBranchOpPattern>(typeConverter, context);
}
