//===- TosaOptimization.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass to perform optimizations on TOSA operations
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/PassDetail.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::tosa;

#define PASS_NAME "tosa-optimization"
#define DEBUG_TYPE PASS_NAME

namespace {

struct Conv2DIsFullyConnected : public OpRewritePattern<tosa::Conv2DOp> {
  explicit Conv2DIsFullyConnected(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(tosa::Conv2DOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.input();
    Value weight = op.weight();
    ShapedType inputType = input.getType().cast<ShapedType>();
    ShapedType weightType = weight.getType().cast<ShapedType>();
    ShapedType resultType = op.getType().cast<ShapedType>();

    if (!inputType.hasStaticShape() || !weightType.hasRank()) {
      return failure();
    }

    // Stride must be 1 for this optimization.
    for (Attribute stride : op.stride().getValue()) {
      if (!stride.cast<IntegerAttr>().getValue().isOne()) {
        return failure();
      }
    }

    // Only works for a 1x1 kernel.
    ArrayRef<int64_t> weightShape = weightType.getShape();
    if (weightShape[1] != 1 || weightShape[2] != 1) {
      return failure();
    }

    // Reshape input to [N,IH,IW,IC] -> [N * IH * IW, IC].
    ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::SmallVector<int64_t, 2> revisedInputShape{
        inputShape[0] * inputShape[1] * inputShape[2], inputShape[3]};
    auto revisedInputShapeType = RankedTensorType::get(
        revisedInputShape,
        input.getType().dyn_cast<RankedTensorType>().getElementType());
    auto reshapedInput = rewriter
                             .create<tosa::ReshapeOp>(
                                 op.getLoc(), revisedInputShapeType, input,
                                 rewriter.getI64ArrayAttr(revisedInputShape))
                             .getResult();

    // Reshape kernel to [OC,KH,KW,IC] -> [OC, IC].
    llvm::SmallVector<int64_t, 2> revisedWeightShape{weightShape[0],
                                                     weightShape[3]};
    auto revisedWeightShapeType = RankedTensorType::get(
        revisedWeightShape,
        weight.getType().dyn_cast<RankedTensorType>().getElementType());
    auto reshapedWeight = rewriter
                              .create<tosa::ReshapeOp>(
                                  op.getLoc(), revisedWeightShapeType, weight,
                                  rewriter.getI64ArrayAttr(revisedWeightShape))
                              .getResult();

    // Perform a fully connected network over the reshaped input and weight.
    llvm::SmallVector<int64_t, 2> fullyConnectedShape{
        inputShape[0] * inputShape[1] * inputShape[2], weightShape[0]};
    auto fullyConnectedShapeType = RankedTensorType::get(
        fullyConnectedShape,
        resultType.dyn_cast<ShapedType>().getElementType());

    Value fullyConnectedValue;
    if (op.quantization_info()) {
      fullyConnectedValue =
          rewriter
              .create<tosa::FullyConnectedOp>(
                  op.getLoc(), fullyConnectedShapeType, reshapedInput,
                  reshapedWeight, op.bias(), op.quantization_info().getValue())
              .getResult();
    } else {
      fullyConnectedValue = rewriter
                                .create<tosa::FullyConnectedOp>(
                                    op.getLoc(), fullyConnectedShapeType,
                                    reshapedInput, reshapedWeight, op.bias())
                                .getResult();
    }

    // Reshape output to [N, IH, IW, OC].
    llvm::SmallVector<int64_t, 4> outputShape{inputShape[0], inputShape[1],
                                              inputShape[2], weightShape[0]};
    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
        op, resultType, fullyConnectedValue,
        rewriter.getI64ArrayAttr(outputShape));
    return success();
  }
};

struct DepthwiseConv2DIsMul : public OpRewritePattern<tosa::DepthwiseConv2DOp> {
  explicit DepthwiseConv2DIsMul(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(tosa::DepthwiseConv2DOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.input();
    Value weight = op.weight();
    ShapedType inputType = input.getType().cast<ShapedType>();
    ShapedType weightType = weight.getType().cast<ShapedType>();
    ShapedType resultType = op.output().getType().cast<ShapedType>();
    Type inputEType = inputType.getElementType();

    if (!(inputType.hasStaticShape() && weightType.hasStaticShape() &&
          resultType.hasStaticShape())) {
      return failure();
    }

    // Quantization information needs to still be performed.
    if (op.quantization_info() || !inputEType.isa<FloatType>()) {
      return failure();
    }

    // Stride must be 1 for this optimization.
    for (Attribute stride : op.stride().getValue()) {
      if (!stride.cast<IntegerAttr>().getValue().isOne()) {
        return failure();
      }
    }

    // Only works for a 1x1 kernel.
    ArrayRef<int64_t> weightShape = weightType.getShape();
    if (weightShape[0] != 1 || weightShape[1] != 1) {
      return failure();
    }

    // Reshape input to [N, H, W, C] -> [N, H, W, C, 1].
    ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::SmallVector<int64_t, 2> revisedInputShape{
        inputShape[0], inputShape[1], inputShape[2], inputShape[3], 1};
    auto revisedInputShapeType = RankedTensorType::get(
        revisedInputShape,
        input.getType().dyn_cast<RankedTensorType>().getElementType());
    auto reshapedInput = rewriter
                             .create<tosa::ReshapeOp>(
                                 op.getLoc(), revisedInputShapeType, input,
                                 rewriter.getI64ArrayAttr(revisedInputShape))
                             .getResult();

    // Reshape kernel to [KH, KW, C, M] -> [1, 1, 1, C, M].
    llvm::SmallVector<int64_t, 2> revisedWeightShape{1, 1, 1, weightShape[2],
                                                     weightShape[3]};
    auto revisedWeightShapeType = RankedTensorType::get(
        revisedWeightShape,
        weight.getType().dyn_cast<RankedTensorType>().getElementType());
    auto reshapedWeight = rewriter
                              .create<tosa::ReshapeOp>(
                                  op.getLoc(), revisedWeightShapeType, weight,
                                  rewriter.getI64ArrayAttr(revisedWeightShape))
                              .getResult();

    // Perform an elementwise mul over the reshaped input and weight.
    llvm::SmallVector<int64_t, 2> mulShape{inputShape[0], inputShape[1],
                                           inputShape[2], inputShape[3],
                                           weightShape[3]};
    auto mulShapeType = RankedTensorType::get(
        mulShape,
        weight.getType().dyn_cast<RankedTensorType>().getElementType());
    Value mulValue =
        rewriter
            .create<tosa::MulOp>(op.getLoc(), mulShapeType, reshapedInput,
                                 reshapedWeight, /*shift=*/0)
            .getResult();

    // Reshape output to [N, H, W, C * M].
    auto outputShape = op.output().getType().cast<ShapedType>().getShape();
    auto outputShapeType = RankedTensorType::get(
        outputShape,
        input.getType().dyn_cast<RankedTensorType>().getElementType());
    auto outputValue =
        rewriter.create<tosa::ReshapeOp>(op.getLoc(), outputShapeType, mulValue,
                                         rewriter.getI64ArrayAttr(outputShape));

    // Add in the bias.
    rewriter
        .replaceOpWithNewOp<tosa::AddOp>(op, outputShapeType, outputValue,
                                         op.bias())
        .getResult();
    return success();
  }
};

class TosaOptimization : public PassWrapper<TosaOptimization, FunctionPass> {
public:
  explicit TosaOptimization() = default;
  void runOnFunction() override;

  StringRef getArgument() const final { return PASS_NAME; }
  StringRef getDescription() const final {
    return "Applies TOSA Operation Optimizations";
  }
};

void TosaOptimization::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());

  patterns.insert<Conv2DIsFullyConnected>(&getContext());
  patterns.insert<DepthwiseConv2DIsMul>(&getContext());

  auto func = getFunction();
  if (applyPatternsAndFoldGreedily(func, std::move(patterns)).failed()) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaOptimizationPass() {
  return std::make_unique<TosaOptimization>();
}
