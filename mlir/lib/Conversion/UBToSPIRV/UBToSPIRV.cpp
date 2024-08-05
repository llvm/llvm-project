//===- UBToSPIRV.cpp - UB to SPIRV-V dialect conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/UBToSPIRV/UBToSPIRV.h"

#include "mlir/Conversion/ConvertToSPIRV/ToSPIRVInterface.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_UBTOSPIRVCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct PoisonOpLowering final : OpConversionPattern<ub::PoisonOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ub::PoisonOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type origType = op.getType();
    if (!origType.isIntOrIndexOrFloat())
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "unsupported type " << origType;
      });

    Type resType = getTypeConverter()->convertType(origType);
    if (!resType)
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "failed to convert result type " << origType;
      });

    rewriter.replaceOpWithNewOp<spirv::UndefOp>(op, resType);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct UBToSPIRVConversionPass final
    : impl::UBToSPIRVConversionPassBase<UBToSPIRVConversionPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    spirv::TargetEnvAttr targetAttr = spirv::lookupTargetEnvOrDefault(op);
    std::unique_ptr<SPIRVConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);

    SPIRVConversionOptions options;
    SPIRVTypeConverter typeConverter(targetAttr, options);

    RewritePatternSet patterns(&getContext());
    ub::populateUBToSPIRVConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, *target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::ub::populateUBToSPIRVConversionPatterns(
    SPIRVTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<PoisonOpLowering>(converter, patterns.getContext());
}

//===----------------------------------------------------------------------===//
// ConvertToSPIRVPatternInterface implementation
//===----------------------------------------------------------------------===//
namespace {
/// Implement the interface to convert ub to SPIR-V.
struct ToSPIRVDialectInterface : public ConvertToSPIRVPatternInterface {
  using ConvertToSPIRVPatternInterface::ConvertToSPIRVPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<spirv::SPIRVDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToSPIRVConversionPatterns(
      ConversionTarget &target, SPIRVTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    ub::populateUBToSPIRVConversionPatterns(typeConverter, patterns);
  }
};
} // namespace

void mlir::ub::registerConvertUBToSPIRVInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ub::UBDialect *dialect) {
    dialect->addInterfaces<ToSPIRVDialectInterface>();
  });
}
