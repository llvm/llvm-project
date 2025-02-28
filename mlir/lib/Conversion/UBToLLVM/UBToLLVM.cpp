//===- UBToLLVM.cpp - UB to LLVM dialect conversion -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_UBTOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct PoisonOpLowering : public ConvertOpToLLVMPattern<ub::PoisonOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ub::PoisonOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// PoisonOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
PoisonOpLowering::matchAndRewrite(ub::PoisonOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  if (!isa<ub::PoisonAttr>(op.getValue())) {
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "pattern can only convert op with '"
           << ub::PoisonAttr::getMnemonic() << "' poison value";
    });
  }

  Type resType = getTypeConverter()->convertType(op.getType());
  if (!resType) {
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "failed to convert result type " << op.getType();
    });
  }

  rewriter.replaceOpWithNewOp<LLVM::PoisonOp>(op, resType);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct UBToLLVMConversionPass
    : public impl::UBToLLVMConversionPassBase<UBToLLVMConversionPass> {
  using Base::Base;

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    LowerToLLVMOptions options(&getContext());
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    LLVMTypeConverter converter(&getContext(), options);
    mlir::ub::populateUBToLLVMConversionPatterns(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::ub::populateUBToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<PoisonOpLowering>(converter);
}

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

namespace {
/// Implement the interface to convert UB to LLVM.
struct UBToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);
  }
};
} // namespace

void mlir::ub::registerConvertUBToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ub::UBDialect *dialect) {
    dialect->addInterfaces<UBToLLVMDialectInterface>();
  });
}
