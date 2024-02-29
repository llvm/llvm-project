//===- VectorLinearize.cpp - vector linearization transforms --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns and pass for linearizing ND vectors into 1D.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct LinearizeConstant final : OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = constOp.getLoc();
    auto resType =
        getTypeConverter()->convertType<VectorType>(constOp.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(loc, "can't convert return type");

    auto dstElementsAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
    if (!dstElementsAttr)
      return rewriter.notifyMatchFailure(loc, "unsupported attr type");

    dstElementsAttr = dstElementsAttr.reshape(resType);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(constOp, resType,
                                                   dstElementsAttr);
    return success();
  }
};

struct LinearizeVectorizable final
    : OpTraitConversionPattern<OpTrait::Vectorizable> {
  using OpTraitConversionPattern::OpTraitConversionPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Operation *> newOp =
        convertOpResultTypes(op, operands, *getTypeConverter(), rewriter);
    if (failed(newOp))
      return failure();

    rewriter.replaceOp(op, (*newOp)->getResults());
    return success();
  }
};
} // namespace

void mlir::vector::populateVectorLinearizeTypeConversionsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  typeConverter.addConversion([](VectorType type) -> std::optional<Type> {
    // Ignore scalable vectors for now.
    if (type.getRank() <= 1 || type.isScalable())
      return type;

    return VectorType::get(type.getNumElements(), type.getElementType());
  });

  auto materializeCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                            Location loc) -> Value {
    if (inputs.size() != 1 || !isa<VectorType>(inputs.front().getType()) ||
        !isa<VectorType>(type))
      return nullptr;

    return builder.create<vector::ShapeCastOp>(loc, type, inputs.front());
  };
  typeConverter.addArgumentMaterialization(materializeCast);
  typeConverter.addSourceMaterialization(materializeCast);
  typeConverter.addTargetMaterialization(materializeCast);

  target.markUnknownOpDynamicallyLegal(
      [&](Operation *op) -> std::optional<bool> {
        if (isa<arith::ConstantOp>(op) || op->hasTrait<OpTrait::Vectorizable>())
          return typeConverter.isLegal(op);

        return std::nullopt;
      });

  patterns.add<LinearizeConstant, LinearizeVectorizable>(typeConverter,
                                                         patterns.getContext());
}
