//===- TosaTestPasses.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test passes to exercise TOSA helper functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/PassDetail.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define PASS_NAME "tosa-test-quant-utils"

using namespace mlir;
using namespace mlir::tosa;

// This transformation converts quantized uint8 to quantized int8. The
// construction of the new type invokes buildQTypeFromMinMax. Extracted from
// TOSA legalization infrastructure.
struct ConvertTosaNegateOp : public RewritePattern {
  explicit ConvertTosaNegateOp(MLIRContext *context)
      : RewritePattern(tosa::NegateOp::getOperationName(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult
ConvertTosaNegateOp::matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const {

  auto tosaNegateOp = cast<tosa::NegateOp>(op);

  auto inputType =
      tosaNegateOp.input1().getType().dyn_cast<mlir::RankedTensorType>();
  // skip if input is not ranked tensor type
  if (!inputType)
    return failure();

  // skip if it's not ranked tensor type.
  auto outputType =
      tosaNegateOp.getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!outputType)
    return failure();

  // skip if output is not per-tensor quantized type.
  auto outputElementType =
      outputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
  if (!outputElementType)
    return failure();

  // skip if output is not uint8.
  if (outputElementType.isSigned() ||
      outputElementType.getStorageTypeIntegralWidth() != 8)
    return failure();

  double typeRangeMin = double(outputElementType.getStorageTypeMin() -
                               outputElementType.getZeroPoint()) *
                        outputElementType.getScale();
  double typeRangeMax = double(outputElementType.getStorageTypeMax() -
                               outputElementType.getZeroPoint()) *
                        outputElementType.getScale();
  bool narrowRange = outputElementType.getStorageTypeMin() == 1;

  auto dstQConstType = RankedTensorType::get(
      outputType.getShape(),
      buildQTypeFromMinMax(rewriter, outputElementType.getExpressedType(),
                           rewriter.getF64FloatAttr(typeRangeMin),
                           rewriter.getF64FloatAttr(typeRangeMax),
                           rewriter.getI32IntegerAttr(
                               outputElementType.getStorageTypeIntegralWidth()),
                           0, true /* signed */,
                           rewriter.getBoolAttr(narrowRange)));

  ElementsAttr inputElems;
  if (!matchPattern(tosaNegateOp.input1(), m_Constant(&inputElems)))
    return failure();

  auto newConstOp =
      rewriter.create<tosa::ConstOp>(op->getLoc(), dstQConstType, inputElems);
  auto newNegateOp = rewriter.create<tosa::NegateOp>(
      op->getLoc(), dstQConstType, newConstOp.getResult());

  rewriter.replaceOp(op, {newNegateOp.getResult()});
  return success();
}

// This transformation modifies the quantized output of a test conv2d input and
// appends a TOSA rescale after it. The rescale op requires the invocation of
// computeMultiplierAndShift. From TOSA legalization infrastructure.
struct ConvertTosaConv2DOp : public RewritePattern {
  explicit ConvertTosaConv2DOp(MLIRContext *context)
      : RewritePattern(tosa::Conv2DOp::getOperationName(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult
ConvertTosaConv2DOp::matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const {

  auto tosaConv2DOp = cast<tosa::Conv2DOp>(op);

  auto inputType =
      tosaConv2DOp.input().getType().dyn_cast<mlir::RankedTensorType>();

  // skip if input is not ranked tensor type
  if (!inputType)
    return failure();

  auto weightType =
      tosaConv2DOp.weight().getType().dyn_cast<mlir::RankedTensorType>();

  // skip if wt is not ranked tensor type
  if (!weightType)
    return failure();

  // skip if it's not ranked tensor type.
  auto outputType =
      tosaConv2DOp.getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!outputType)
    return failure();

  auto inputQType =
      inputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
  auto weightQType =
      weightType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
  auto outputQType =
      outputType.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();

  // Works on quantized type only.
  if (!(inputQType && weightQType && outputQType))
    return failure();

  auto newTosaConv2DOpType =
      RankedTensorType::get(outputType.getShape(), rewriter.getIntegerType(32));

  auto newTosaConv2DOp = rewriter.create<tosa::Conv2DOp>(
      op->getLoc(), newTosaConv2DOpType, tosaConv2DOp.input(),
      tosaConv2DOp.weight(), tosaConv2DOp.bias(), tosaConv2DOp.pad(),
      tosaConv2DOp.stride(), tosaConv2DOp.dilation());

  // Create rescale to quantized type
  double inputScale = inputQType.getScale();
  double weightScale = weightQType.getScale();
  double outputScale = outputQType.getScale();
  int64_t outputZp = outputQType.getZeroPoint();

  double opTensorScale = (inputScale * weightScale) / outputScale;

  int32_t multiplier;
  int32_t shift;

  // Obtain the quantized scale = multiplier and shift.
  computeMultiplierAndShift(opTensorScale, multiplier, shift, 32);

  auto newTosaRescaleOp = rewriter.create<tosa::RescaleOp>(
      op->getLoc(), outputType, newTosaConv2DOp.getResult(),
      rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(outputZp),
      rewriter.getI32ArrayAttr({multiplier}), rewriter.getI32ArrayAttr({shift}),
      rewriter.getBoolAttr(true), rewriter.getBoolAttr(true),
      rewriter.getBoolAttr(false));

  rewriter.replaceOp(op, {newTosaRescaleOp.getResult()});
  return success();
}

namespace {

struct TosaTestQuantUtilAPI
    : public PassWrapper<TosaTestQuantUtilAPI, OperationPass<FuncOp>> {
  StringRef getArgument() const final { return PASS_NAME; }
  StringRef getDescription() const final {
    return "TOSA Test: Exercise the APIs in QuantUtils.cpp.";
  }
  void runOnOperation() override;
};

void TosaTestQuantUtilAPI::runOnOperation() {
  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  auto func = getOperation();

  patterns.add<ConvertTosaNegateOp>(ctx);
  patterns.add<ConvertTosaConv2DOp>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

} // namespace

namespace mlir {
void registerTosaTestQuantUtilAPIPass() {
  PassRegistration<TosaTestQuantUtilAPI>();
}
} // namespace mlir
