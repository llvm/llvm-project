//===- LinalgToStandard.cpp - conversion from Linalg to Standard dialect --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"

#include "../PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace {

/// Conversion patterns.
class SizeToIndexOpConversion
    : public OpConversionPattern<shape::SizeToIndexOp> {
public:
  using OpConversionPattern<shape::SizeToIndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(shape::SizeToIndexOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    shape::SizeToIndexOpOperandAdaptor transformed(operands);
    rewriter.replaceOp(op.getOperation(), transformed.arg());
    return success();
  }
};

class IndexToSizeOpConversion
    : public OpConversionPattern<shape::IndexToSizeOp> {
public:
  using OpConversionPattern<shape::IndexToSizeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(shape::IndexToSizeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    shape::IndexToSizeOpOperandAdaptor transformed(operands);
    rewriter.replaceOp(op.getOperation(), transformed.arg());
    return success();
  }
};

/// Type conversions.
class ShapeTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;

  ShapeTypeConverter(MLIRContext *ctx) {
    // Add default pass-through conversion.
    addConversion([&](Type type) { return type; });
    addConversion([ctx](shape::SizeType type) { return IndexType::get(ctx); });
  }
};

/// Conversion pass.
class ConvertShapeToStandardPass
    : public ConvertShapeToStandardBase<ConvertShapeToStandardPass> {

  void runOnOperation() override {

    // Setup type conversion.
    MLIRContext &ctx = getContext();
    ShapeTypeConverter typeConverter(&ctx);

    // Setup target legality.
    ConversionTarget target(ctx);
    target.addLegalDialect<scf::SCFDialect, StandardOpsDialect>();
    target.addLegalOp<ModuleOp, ModuleTerminatorOp, ReturnOp>();
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });

    // Setup conversion patterns.
    OwningRewritePatternList patterns;
    populateShapeToStandardConversionPatterns(patterns, &ctx);
    populateFuncOpTypeConversionPattern(patterns, &ctx, typeConverter);

    // Apply conversion.
    auto module = getOperation();
    if (failed(applyFullConversion(module, target, patterns, &typeConverter)))
      signalPassFailure();
  }
};

} // namespace

void populateShapeToStandardConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  // clang-format off
  patterns.insert<
      IndexToSizeOpConversion,
      SizeToIndexOpConversion>(ctx);
  // clang-format on
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertShapeToStandardPass() {
  return std::make_unique<ConvertShapeToStandardPass>();
}

} // namespace mlir
