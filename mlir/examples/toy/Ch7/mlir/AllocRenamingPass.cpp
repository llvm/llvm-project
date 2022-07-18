//====- AllocRenamingPass.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the renaming of '_mlir_alloc' and '_mlir_free' functions
// respectively into 'malloc' and 'free', so that the Toy example doesn't have
// to deal with runtime libraries to be linked.
//
//===----------------------------------------------------------------------===//

#include "toy/Passes.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AllocRenamingPass RewritePatterns
//===----------------------------------------------------------------------===//

namespace {
/// Rename the '_mlir_alloc' function into 'malloc'
class AllocFuncRenamePattern : public OpRewritePattern<LLVM::LLVMFuncOp> {
public:
  using OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;

  LogicalResult match(LLVM::LLVMFuncOp op) const override {
    return LogicalResult::success(op.getName() == "_mlir_alloc");
  }

  void rewrite(LLVM::LLVMFuncOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::LLVMFuncOp>(
        op, "malloc", op.getFunctionType(), op.getLinkage());
  }
};

/// Rename the '_mlir_free' function into 'free'
class FreeFuncRenamePattern : public OpRewritePattern<LLVM::LLVMFuncOp> {
public:
  using OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;

  LogicalResult match(LLVM::LLVMFuncOp op) const override {
    return LogicalResult::success(op.getName() == "_mlir_free");
  }

  void rewrite(LLVM::LLVMFuncOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::LLVMFuncOp>(
        op, "free", op.getFunctionType(), op.getLinkage());
  }
};

/// Rename the calls to '_mlir_alloc' with calls to 'malloc'
class AllocCallRenamePattern : public OpRewritePattern<LLVM::CallOp> {
public:
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult match(LLVM::CallOp op) const override {
    auto callee = op.getCallee();

    if (!callee)
      return failure();

    return LogicalResult::success(*callee == "_mlir_alloc");
  }

  void rewrite(LLVM::CallOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, op.getResultTypes(), "malloc",
                                              op.getOperands());
  }
};

/// Rename the calls to '_mlir_free' with calls to 'free'
class FreeCallRenamePattern : public OpRewritePattern<LLVM::CallOp> {
public:
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult match(LLVM::CallOp op) const override {
    auto callee = op.getCallee();

    if (!callee)
      return failure();

    return LogicalResult::success(*callee == "_mlir_free");
  }

  void rewrite(LLVM::CallOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, op.getResultTypes(), "free",
                                              op.getOperands());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// AllocRenamingPass
//===----------------------------------------------------------------------===//

namespace {
struct AllocRenamingPass
    : public PassWrapper<AllocRenamingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AllocRenamingPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void AllocRenamingPass::runOnOperation() {
  LLVMConversionTarget target(getContext());

  target.addDynamicallyLegalOp<LLVM::LLVMFuncOp>([](LLVM::LLVMFuncOp op) {
    auto name = op.getName();
    return name != "_mlir_alloc" && name != "_mlir_free";
  });

  target.addDynamicallyLegalOp<LLVM::CallOp>([](LLVM::CallOp op) {
    auto callee = op.getCallee();

    if (!callee)
      return true;

    return *callee != "_mlir_alloc" && *callee != "_mlir_free";
  });

  target.markUnknownOpDynamicallyLegal(
      [](mlir::Operation *op) { return true; });

  RewritePatternSet patterns(&getContext());

  patterns.add<AllocFuncRenamePattern>(&getContext());
  patterns.add<FreeFuncRenamePattern>(&getContext());
  patterns.add<AllocCallRenamePattern>(&getContext());
  patterns.add<FreeCallRenamePattern>(&getContext());

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass to rename the '_mlir_alloc' and '_mlir_free' functions to
/// 'malloc' and 'free'.
std::unique_ptr<mlir::Pass> mlir::toy::createAllocRenamingPass() {
  return std::make_unique<AllocRenamingPass>();
}
