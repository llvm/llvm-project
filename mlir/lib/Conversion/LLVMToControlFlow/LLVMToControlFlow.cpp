//===- LLVMToControlFlow.cpp - LLVM to CF conversion ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMToControlFlow/LLVMToControlFlow.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLLVMTOCONTROLFLOW
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct BranchOpPattern : public OpConversionPattern<LLVM::BrOp> {
  using OpConversionPattern<LLVM::BrOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::BrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, adaptor.getOperands(),
                                              op.getSuccessor());
    return success();
  }
};

struct CondBranchOpPattern : public OpConversionPattern<LLVM::CondBrOp> {
  using OpConversionPattern<LLVM::CondBrOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::CondBrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    return success();
  }
};

struct ConvertLLVMToControlFlowPass
    : public impl::ConvertLLVMToControlFlowBase<ConvertLLVMToControlFlowPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertLLVMToControlFlowPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<BranchOpPattern, CondBranchOpPattern>(&getContext());
  // Configure conversion to lower out SCF operations.
  ConversionTarget target(getContext());
  target.addIllegalOp<LLVM::BrOp, LLVM::CondBrOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
