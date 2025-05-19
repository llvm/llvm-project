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

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/PrintCallHelper.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"
#include <functional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTCONTROLFLOWTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define PASS_NAME "convert-cf-to-llvm"

namespace {
/// Lower `cf.assert`. The default lowering calls the `abort` function if the
/// assertion is violated and has no effect otherwise. The failure message is
/// ignored by the default lowering but should be propagated by any custom
/// lowering.
struct AssertOpLowering : public ConvertOpToLLVMPattern<cf::AssertOp> {
  explicit AssertOpLowering(const LLVMTypeConverter &typeConverter,
                            bool abortOnFailedAssert = true)
      : ConvertOpToLLVMPattern<cf::AssertOp>(typeConverter, /*benefit=*/1),
        abortOnFailedAssert(abortOnFailedAssert) {}

  LogicalResult
  matchAndRewrite(cf::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    // Split block at `assert` operation.
    Block *opBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    Block *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

    // Failed block: Generate IR to print the message and call `abort`.
    Block *failureBlock = rewriter.createBlock(opBlock->getParent());
    auto createResult = LLVM::createPrintStrCall(
        rewriter, loc, module, "assert_msg", op.getMsg(), *getTypeConverter(),
        /*addNewLine=*/false,
        /*runtimeFunctionName=*/"puts");
    if (createResult.failed())
      return failure();

    if (abortOnFailedAssert) {
      // Insert the `abort` declaration if necessary.
      auto abortFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("abort");
      if (!abortFunc) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        auto abortFuncTy = LLVM::LLVMFunctionType::get(getVoidType(), {});
        abortFunc = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                      "abort", abortFuncTy);
      }
      rewriter.create<LLVM::CallOp>(loc, abortFunc, std::nullopt);
      rewriter.create<LLVM::UnreachableOp>(loc);
    } else {
      rewriter.create<LLVM::BrOp>(loc, ValueRange(), continuationBlock);
    }

    // Generate assertion test.
    rewriter.setInsertionPointToEnd(opBlock);
    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
        op, adaptor.getArg(), continuationBlock, failureBlock);

    return success();
  }

private:
  /// If set to `false`, messages are printed but program execution continues.
  /// This is useful for testing asserts.
  bool abortOnFailedAssert = true;
};

/// Helper function for converting branch ops. This function converts the
/// signature of the given block. If the new block signature is different from
/// `expectedTypes`, returns "failure".
static FailureOr<Block *> getConvertedBlock(ConversionPatternRewriter &rewriter,
                                            const TypeConverter *converter,
                                            Operation *branchOp, Block *block,
                                            TypeRange expectedTypes) {
  assert(converter && "expected non-null type converter");
  assert(!block->isEntryBlock() && "entry blocks have no predecessors");

  // There is nothing to do if the types already match.
  if (block->getArgumentTypes() == expectedTypes)
    return block;

  // Compute the new block argument types and convert the block.
  std::optional<TypeConverter::SignatureConversion> conversion =
      converter->convertBlockSignature(block);
  if (!conversion)
    return rewriter.notifyMatchFailure(branchOp,
                                       "could not compute block signature");
  if (expectedTypes != conversion->getConvertedTypes())
    return rewriter.notifyMatchFailure(
        branchOp,
        "mismatch between adaptor operand types and computed block signature");
  return rewriter.applySignatureConversion(block, *conversion, converter);
}

/// Convert the destination block signature (if necessary) and lower the branch
/// op to llvm.br.
struct BranchOpLowering : public ConvertOpToLLVMPattern<cf::BranchOp> {
  using ConvertOpToLLVMPattern<cf::BranchOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, typename cf::BranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Block *> convertedBlock =
        getConvertedBlock(rewriter, getTypeConverter(), op, op.getSuccessor(),
                          TypeRange(adaptor.getOperands()));
    if (failed(convertedBlock))
      return failure();
    Operation *newOp = rewriter.replaceOpWithNewOp<LLVM::BrOp>(
        op, adaptor.getOperands(), *convertedBlock);
    // TODO: We should not just forward all attributes like that. But there are
    // existing Flang tests that depend on this behavior.
    newOp->setAttrs(op->getAttrDictionary());
    return success();
  }
};

/// Convert the destination block signatures (if necessary) and lower the
/// branch op to llvm.cond_br.
struct CondBranchOpLowering : public ConvertOpToLLVMPattern<cf::CondBranchOp> {
  using ConvertOpToLLVMPattern<cf::CondBranchOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op,
                  typename cf::CondBranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Block *> convertedTrueBlock =
        getConvertedBlock(rewriter, getTypeConverter(), op, op.getTrueDest(),
                          TypeRange(adaptor.getTrueDestOperands()));
    if (failed(convertedTrueBlock))
      return failure();
    FailureOr<Block *> convertedFalseBlock =
        getConvertedBlock(rewriter, getTypeConverter(), op, op.getFalseDest(),
                          TypeRange(adaptor.getFalseDestOperands()));
    if (failed(convertedFalseBlock))
      return failure();
    Operation *newOp = rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
        op, adaptor.getCondition(), *convertedTrueBlock,
        adaptor.getTrueDestOperands(), *convertedFalseBlock,
        adaptor.getFalseDestOperands());
    // TODO: We should not just forward all attributes like that. But there are
    // existing Flang tests that depend on this behavior.
    newOp->setAttrs(op->getAttrDictionary());
    return success();
  }
};

/// Convert the destination block signatures (if necessary) and lower the
/// switch op to llvm.switch.
struct SwitchOpLowering : public ConvertOpToLLVMPattern<cf::SwitchOp> {
  using ConvertOpToLLVMPattern<cf::SwitchOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cf::SwitchOp op, typename cf::SwitchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get or convert default block.
    FailureOr<Block *> convertedDefaultBlock = getConvertedBlock(
        rewriter, getTypeConverter(), op, op.getDefaultDestination(),
        TypeRange(adaptor.getDefaultOperands()));
    if (failed(convertedDefaultBlock))
      return failure();

    // Get or convert all case blocks.
    SmallVector<Block *> caseDestinations;
    SmallVector<ValueRange> caseOperands = adaptor.getCaseOperands();
    for (auto it : llvm::enumerate(op.getCaseDestinations())) {
      Block *b = it.value();
      FailureOr<Block *> convertedBlock =
          getConvertedBlock(rewriter, getTypeConverter(), op, b,
                            TypeRange(caseOperands[it.index()]));
      if (failed(convertedBlock))
        return failure();
      caseDestinations.push_back(*convertedBlock);
    }

    rewriter.replaceOpWithNewOp<LLVM::SwitchOp>(
        op, adaptor.getFlag(), *convertedDefaultBlock,
        adaptor.getDefaultOperands(), adaptor.getCaseValuesAttr(),
        caseDestinations, caseOperands);
    return success();
  }
};

} // namespace

void mlir::cf::populateControlFlowToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      BranchOpLowering,
      CondBranchOpLowering,
      SwitchOpLowering>(converter);
  // clang-format on
}

void mlir::cf::populateAssertToLLVMConversionPattern(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    bool abortOnFailure) {
  patterns.add<AssertOpLowering>(converter, abortOnFailure);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
/// A pass converting MLIR operations into the LLVM IR dialect.
struct ConvertControlFlowToLLVM
    : public impl::ConvertControlFlowToLLVMPassBase<ConvertControlFlowToLLVM> {

  using Base::Base;

  /// Run the dialect converter on the module.
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    LLVMConversionTarget target(*ctx);
    // This pass lowers only CF dialect ops, but it also modifies block
    // signatures inside other ops. These ops should be treated as legal. They
    // are lowered by other passes.
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return op->getDialect() !=
             ctx->getLoadedDialect<cf::ControlFlowDialect>();
    });

    LowerToLLVMOptions options(ctx);
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    LLVMTypeConverter converter(ctx, options);
    RewritePatternSet patterns(ctx);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    mlir::cf::populateAssertToLLVMConversionPattern(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

namespace {
/// Implement the interface to convert MemRef to LLVM.
struct ControlFlowToLLVMDialectInterface
    : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::cf::populateAssertToLLVMConversionPattern(typeConverter, patterns);
  }
};
} // namespace

void mlir::cf::registerConvertControlFlowToLLVMInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, cf::ControlFlowDialect *dialect) {
    dialect->addInterfaces<ControlFlowToLLVMDialectInterface>();
  });
}
