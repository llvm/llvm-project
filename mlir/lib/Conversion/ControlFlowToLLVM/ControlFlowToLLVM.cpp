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

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
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
#define GEN_PASS_DEF_CONVERTCONTROLFLOWTOLLVM
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define PASS_NAME "convert-cf-to-llvm"

static std::string generateGlobalMsgSymbolName(ModuleOp moduleOp) {
  std::string prefix = "assert_msg_";
  int counter = 0;
  while (moduleOp.lookupSymbol(prefix + std::to_string(counter)))
    ++counter;
  return prefix + std::to_string(counter);
}

/// Generate IR that prints the given string to stderr.
static void createPrintMsg(OpBuilder &builder, Location loc, ModuleOp moduleOp,
                           StringRef msg) {
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(moduleOp.getBody());
  MLIRContext *ctx = builder.getContext();

  // Create a zero-terminated byte representation and allocate global symbol.
  SmallVector<uint8_t> elementVals;
  elementVals.append(msg.begin(), msg.end());
  elementVals.push_back(0);
  auto dataAttrType = RankedTensorType::get(
      {static_cast<int64_t>(elementVals.size())}, builder.getI8Type());
  auto dataAttr =
      DenseElementsAttr::get(dataAttrType, llvm::ArrayRef(elementVals));
  auto arrayTy =
      LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8), elementVals.size());
  std::string symbolName = generateGlobalMsgSymbolName(moduleOp);
  auto globalOp = builder.create<LLVM::GlobalOp>(
      loc, arrayTy, /*constant=*/true, LLVM::Linkage::Private, symbolName,
      dataAttr);

  // Emit call to `printStr` in runtime library.
  builder.restoreInsertionPoint(ip);
  auto msgAddr = builder.create<LLVM::AddressOfOp>(
      loc, LLVM::LLVMPointerType::get(arrayTy), globalOp.getName());
  SmallVector<LLVM::GEPArg> indices(1, 0);
  Value gep = builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(builder.getI8Type()), msgAddr, indices);
  Operation *printer =
      LLVM::lookupOrCreatePrintStrFn(moduleOp, /*TODO: opaquePointers=*/false);
  builder.create<LLVM::CallOp>(loc, TypeRange(), SymbolRefAttr::get(printer),
                               gep);
}

namespace {
/// Lower `cf.assert`. The default lowering calls the `abort` function if the
/// assertion is violated and has no effect otherwise. The failure message is
/// ignored by the default lowering but should be propagated by any custom
/// lowering.
struct AssertOpLowering : public ConvertOpToLLVMPattern<cf::AssertOp> {
  explicit AssertOpLowering(LLVMTypeConverter &typeConverter,
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
    createPrintMsg(rewriter, loc, module, op.getMsg());
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

/// The cf->LLVM lowerings for branching ops require that the blocks they jump
/// to first have updated types which should be handled by a pattern operating
/// on the parent op.
static LogicalResult verifyMatchingValues(ConversionPatternRewriter &rewriter,
                                          ValueRange operands,
                                          ValueRange blockArgs, Location loc,
                                          llvm::StringRef messagePrefix) {
  for (const auto &idxAndTypes :
       llvm::enumerate(llvm::zip(blockArgs, operands))) {
    int64_t i = idxAndTypes.index();
    Value argValue =
        rewriter.getRemappedValue(std::get<0>(idxAndTypes.value()));
    Type operandType = std::get<1>(idxAndTypes.value()).getType();
    // In the case of an invalid jump, the block argument will have been
    // remapped to an UnrealizedConversionCast. In the case of a valid jump,
    // there might still be a no-op conversion cast with both types being equal.
    // Consider both of these details to see if the jump would be invalid.
    if (auto op = dyn_cast_or_null<UnrealizedConversionCastOp>(
            argValue.getDefiningOp())) {
      if (op.getOperandTypes().front() != operandType) {
        return rewriter.notifyMatchFailure(loc, [&](Diagnostic &diag) {
          diag << messagePrefix;
          diag << "mismatched types from operand # " << i << " ";
          diag << operandType;
          diag << " not compatible with destination block argument type ";
          diag << op.getOperandTypes().front();
          diag << " which should be converted with the parent op.";
        });
      }
    }
  }
  return success();
}

/// Ensure that all block types were updated and then create an LLVM::BrOp
struct BranchOpLowering : public ConvertOpToLLVMPattern<cf::BranchOp> {
  using ConvertOpToLLVMPattern<cf::BranchOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, typename cf::BranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyMatchingValues(rewriter, adaptor.getDestOperands(),
                                    op.getSuccessor()->getArguments(),
                                    op.getLoc(),
                                    /*messagePrefix=*/"")))
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::BrOp>(
        op, adaptor.getOperands(), op->getSuccessors(), op->getAttrs());
    return success();
  }
};

/// Ensure that all block types were updated and then create an LLVM::CondBrOp
struct CondBranchOpLowering : public ConvertOpToLLVMPattern<cf::CondBranchOp> {
  using ConvertOpToLLVMPattern<cf::CondBranchOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op,
                  typename cf::CondBranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyMatchingValues(rewriter, adaptor.getFalseDestOperands(),
                                    op.getFalseDest()->getArguments(),
                                    op.getLoc(), "in false case branch ")))
      return failure();
    if (failed(verifyMatchingValues(rewriter, adaptor.getTrueDestOperands(),
                                    op.getTrueDest()->getArguments(),
                                    op.getLoc(), "in true case branch ")))
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
        op, adaptor.getOperands(), op->getSuccessors(), op->getAttrs());
    return success();
  }
};

/// Ensure that all block types were updated and then create an LLVM::SwitchOp
struct SwitchOpLowering : public ConvertOpToLLVMPattern<cf::SwitchOp> {
  using ConvertOpToLLVMPattern<cf::SwitchOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cf::SwitchOp op, typename cf::SwitchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyMatchingValues(rewriter, adaptor.getDefaultOperands(),
                                    op.getDefaultDestination()->getArguments(),
                                    op.getLoc(), "in switch default case ")))
      return failure();

    for (const auto &i : llvm::enumerate(
             llvm::zip(adaptor.getCaseOperands(), op.getCaseDestinations()))) {
      if (failed(verifyMatchingValues(
              rewriter, std::get<0>(i.value()),
              std::get<1>(i.value())->getArguments(), op.getLoc(),
              "in switch case " + std::to_string(i.index()) + " "))) {
        return failure();
      }
    }

    rewriter.replaceOpWithNewOp<LLVM::SwitchOp>(
        op, adaptor.getOperands(), op->getSuccessors(), op->getAttrs());
    return success();
  }
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

void mlir::cf::populateAssertToLLVMConversionPattern(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    bool abortOnFailure) {
  patterns.add<AssertOpLowering>(converter, abortOnFailure);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
/// A pass converting MLIR operations into the LLVM IR dialect.
struct ConvertControlFlowToLLVM
    : public impl::ConvertControlFlowToLLVMBase<ConvertControlFlowToLLVM> {
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
