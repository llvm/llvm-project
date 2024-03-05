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

static bool isLessThanTargetBitWidth(Operation *op, unsigned targetBitWidth) {
  auto resultTypes = op->getResultTypes();
  for (auto resType : resultTypes) {
    VectorType vecType = cast<VectorType>(resType);
    // Reject index since getElementTypeBitWidth will abort for Index types.
    if (vecType.getElementType().isIndex())
      return false;
    unsigned trailingVecDimBitWidth =
        vecType.getShape().back() * vecType.getElementTypeBitWidth();
    if (trailingVecDimBitWidth >= targetBitWidth)
      return false;
  }
  return true;
}

namespace {
struct LinearizeConstant final : OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LinearizeConstant(
      const TypeConverter &typeConverter, MLIRContext *context,
      unsigned targetVectBitWidth = std::numeric_limits<unsigned>::max(),
      PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        targetVectorBitWidth(targetVectBitWidth) {}
  LogicalResult
  matchAndRewrite(arith::ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = constOp.getLoc();
    auto resType =
        getTypeConverter()->convertType<VectorType>(constOp.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(loc, "can't convert return type");
    if (!isLessThanTargetBitWidth(constOp, targetVectorBitWidth))
      return rewriter.notifyMatchFailure(
          loc, "Can't flatten since targetBitWidth <= OpSize");
    auto dstElementsAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
    if (!dstElementsAttr)
      return rewriter.notifyMatchFailure(loc, "unsupported attr type");

    dstElementsAttr = dstElementsAttr.reshape(resType);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(constOp, resType,
                                                   dstElementsAttr);
    return success();
  }

private:
  unsigned targetVectorBitWidth;
};

struct LinearizeVectorizable final
    : OpTraitConversionPattern<OpTrait::Vectorizable> {
  using OpTraitConversionPattern::OpTraitConversionPattern;

public:
  LinearizeVectorizable(
      const TypeConverter &typeConverter, MLIRContext *context,
      unsigned targetVectBitWidth = std::numeric_limits<unsigned>::max(),
      PatternBenefit benefit = 1)
      : OpTraitConversionPattern(typeConverter, context, benefit),
        targetVectorBitWidth(targetVectBitWidth) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isLessThanTargetBitWidth(op, targetVectorBitWidth))
      return rewriter.notifyMatchFailure(
          op->getLoc(), "Can't flatten since targetBitWidth <= OpSize");
    FailureOr<Operation *> newOp =
        convertOpResultTypes(op, operands, *getTypeConverter(), rewriter);
    if (failed(newOp))
      return failure();

    rewriter.replaceOp(op, (*newOp)->getResults());
    return success();
  }

private:
  unsigned targetVectorBitWidth;
};
} // namespace

void mlir::vector::populateVectorLinearizeTypeConversionsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, unsigned targetBitWidth) {

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
      [=](Operation *op) -> std::optional<bool> {
        if ((isa<arith::ConstantOp>(op) ||
             op->hasTrait<OpTrait::Vectorizable>())) {
          return (isLessThanTargetBitWidth(op, targetBitWidth)
                      ? typeConverter.isLegal(op)
                      : true);
        }
        return std::nullopt;
      });

  patterns.add<LinearizeConstant, LinearizeVectorizable>(
      typeConverter, patterns.getContext(), targetBitWidth);
}
