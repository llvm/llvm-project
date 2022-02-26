//===- ControlFlowToLLVM.cpp - ControlFlow to LLVM dialect conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert MLIR standard and builtin dialects
// into the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include <functional>

using namespace mlir;

#define PASS_NAME "convert-cf-to-llvm"

namespace {
/// Lower `cf.assert`. The default lowering calls the `abort` function if the
/// assertion is violated and has no effect otherwise. The failure message is
/// ignored by the default lowering but should be propagated by any custom
/// lowering.
struct AssertOpLowering : public ConvertOpToLLVMPattern<cf::AssertOp> {
  using ConvertOpToLLVMPattern<cf::AssertOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cf::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Insert the `abort` declaration if necessary.
    auto module = op->getParentOfType<ModuleOp>();
    auto abortFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("abort");
    if (!abortFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto abortFuncTy = LLVM::LLVMFunctionType::get(getVoidType(), {});
      abortFunc = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                    "abort", abortFuncTy);
    }

    // Split block at `assert` operation.
    Block *opBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    Block *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

    // Generate IR to call `abort`.
    Block *failureBlock = rewriter.createBlock(opBlock->getParent());
    rewriter.create<LLVM::CallOp>(loc, abortFunc, llvm::None);
    rewriter.create<LLVM::UnreachableOp>(loc);

    // Generate assertion test.
    rewriter.setInsertionPointToEnd(opBlock);
    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
        op, adaptor.getArg(), continuationBlock, failureBlock);

    return success();
  }
};

// Base class for LLVM IR lowering terminator operations with successors.
template <typename SourceOp, typename TargetOp>
struct OneToOneLLVMTerminatorLowering
    : public ConvertOpToLLVMPattern<SourceOp> {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using Base = OneToOneLLVMTerminatorLowering<SourceOp, TargetOp>;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getOperands(),
                                          op->getSuccessors(), op->getAttrs());
    return success();
  }
};

// FIXME: this should be tablegen'ed as well.
struct BranchOpLowering
    : public OneToOneLLVMTerminatorLowering<cf::BranchOp, LLVM::BrOp> {
  using Base::Base;
};
struct CondBranchOpLowering
    : public OneToOneLLVMTerminatorLowering<cf::CondBranchOp, LLVM::CondBrOp> {
  using Base::Base;
};
struct SwitchOpLowering
    : public OneToOneLLVMTerminatorLowering<cf::SwitchOp, LLVM::SwitchOp> {
  using Base::Base;
};

} // namespace

void mlir::cf::populateControlFlowToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      AssertOpLowering,
      BranchOpLowering,
      CondBranchOpLowering,
      SwitchOpLowering>(converter);
  // clang-format on
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
/// A pass converting MLIR operations into the LLVM IR dialect.
struct ConvertControlFlowToLLVM
    : public ConvertControlFlowToLLVMBase<ConvertControlFlowToLLVM> {
  ConvertControlFlowToLLVM() = default;

  /// Run the dialect converter on the module.
  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    LowerToLLVMOptions options(&getContext());
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    LLVMTypeConverter converter(&getContext(), options);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::cf::createConvertControlFlowToLLVMPass() {
  return std::make_unique<ConvertControlFlowToLLVM>();
}
