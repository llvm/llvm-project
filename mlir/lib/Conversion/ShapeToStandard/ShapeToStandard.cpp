//===- ShapeToStandard.cpp - conversion from Shape to Standard dialect ----===//
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

using namespace mlir;
using namespace mlir::shape;

namespace {

/// Generated conversion patterns.
#include "ShapeToStandardPatterns.inc"

/// Conversion patterns.
template <typename SrcOpTy, typename DstOpTy>
class BinaryOpConversion : public OpConversionPattern<SrcOpTy> {
public:
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOpTy op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<DstOpTy>(op.getOperation(), adaptor.lhs(),
                                         adaptor.rhs());
    return success();
  }
};

class IndexToSizeOpConversion : public OpConversionPattern<IndexToSizeOp> {
public:
  using OpConversionPattern<IndexToSizeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IndexToSizeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    IndexToSizeOp::Adaptor transformed(operands);
    rewriter.replaceOp(op.getOperation(), transformed.arg());
    return success();
  }
};

class SizeToIndexOpConversion : public OpConversionPattern<SizeToIndexOp> {
public:
  using OpConversionPattern<SizeToIndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SizeToIndexOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SizeToIndexOp::Adaptor transformed(operands);
    rewriter.replaceOp(op.getOperation(), transformed.arg());
    return success();
  }
};

class ConstSizeOpConverter : public OpConversionPattern<ConstSizeOp> {
public:
  using OpConversionPattern<ConstSizeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstSizeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ConstantIndexOp>(op.getOperation(),
                                                 op.value().getSExtValue());
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

    addConversion([ctx](SizeType type) { return IndexType::get(ctx); });
    addConversion([ctx](ShapeType type) {
      return RankedTensorType::get({ShapedType::kDynamicSize},
                                   IndexType::get(ctx));
    });
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

void mlir::populateShapeToStandardConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  populateWithGenerated(ctx, &patterns);
  // clang-format off
  patterns.insert<
      BinaryOpConversion<AddOp, AddIOp>,
      BinaryOpConversion<MulOp, MulIOp>,
      ConstSizeOpConverter,
      IndexToSizeOpConversion,
      SizeToIndexOpConversion>(ctx);
  // clang-format on
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertShapeToStandardPass() {
  return std::make_unique<ConvertShapeToStandardPass>();
}
