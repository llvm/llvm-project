//===- TosaMakeBroadcastable.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Insert reshape to binary op's input if needed to match rank
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR//TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/PassDetail.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tosa;

/// There are two potential ways implementing broadcast:
/// a. https://www.tensorflow.org/xla/broadcasting#formal_definition
/// b. https://numpy.org/doc/stable/user/basics.broadcasting.html
/// TBD: picking option (a) now.

/// In this pass, we insert RESHAPE operators to increase the rank of the
/// lower rank operand as a first step in the broadcasting process. The TOSA
/// operators that support broadcast require that the rank of the operands
/// are equal.

// Examples:
// If lower=[a], target=[a, b, c], [a] reshaped into [a, 1, 1].
// TODO: If lower=[b], target=[a, b, c], [b] should but NOT YET reshaped into
// [1, b, 1].
// If lower=[c], target=[a, b, c], [c] reshaped into [1, 1, c].
// If lower=[a, c], target=[a, b, c], [a, c] reshaped into [a, 1, c].
// If lower=[a, b], target=[a, b, c], [a, b] reshaped into [a, b, 1].
// If lower=[b, c], target=[a, b, c], [b, c] reshaped into [1, b, c].
// If lower=[a], target=[a, a], [a] reshaped into [1, a] instead of [a, 1].
// If lower=[a], target=[a, b, a], [a] reshaped into [1, 1, a].
// If lower=[], target=[a, b, c], [] reshaped into [1, 1, 1].

static void computeReshapeOutput(ArrayRef<int64_t> higherRankShape,
                                 ArrayRef<int64_t> lowerRankShape,
                                 SmallVectorImpl<int64_t> &reshapeOutputShape) {
  // Initialize new shapes with [1] * higherRank.
  int64_t higherRank = higherRankShape.size();
  int64_t lowerRank = lowerRankShape.size();

  reshapeOutputShape.assign(higherRank, 1);

  int64_t higherLeftIndex = 0;
  int64_t higherRightIndex = higherRank;
  int64_t lowerLeftIndex = 0;
  int64_t lowerRightIndex = lowerRank;
  int64_t higherRankDim, lowerRankDim;

  if (lowerRightIndex != 0 && higherRightIndex != 0) {
    // Matches lower rank shape from right dimension first, until not
    // matching high rank shape or reaching dimension 0.
    while (true) {
      higherRankDim = higherRankShape[higherRightIndex - 1];
      lowerRankDim = lowerRankShape[lowerRightIndex - 1];
      if (higherRankDim != lowerRankDim)
        break;

      reshapeOutputShape[higherRightIndex - 1] = higherRankDim;

      if (higherRightIndex > 0)
        higherRightIndex--;

      if (lowerRightIndex > 0)
        lowerRightIndex--;

      if (higherRightIndex == 0 || lowerRightIndex == 0)
        break;
    }
    if (lowerRightIndex != 0 && higherRightIndex != 0) {
      // Matches lower rank shape from left dimension, until not matching
      // high rank shape or reaching right index.
      while (true) {
        higherRankDim = higherRankShape[higherLeftIndex];
        lowerRankDim = lowerRankShape[lowerLeftIndex];
        if (higherRankDim != lowerRankDim)
          break;

        reshapeOutputShape[higherLeftIndex] = higherRankDim;

        if (higherLeftIndex < higherRightIndex)
          higherLeftIndex++;

        if (lowerLeftIndex < lowerRightIndex)
          lowerLeftIndex++;

        if (higherLeftIndex == higherRightIndex ||
            lowerLeftIndex == lowerRightIndex)
          break;
      }
    }
  }
}

/// Common code to create the reshape op where necessary to make the rank of the
/// operations equal. Returns the updated input1 and input2 for the original
/// input. The caller is expected to use these to rewrite the original operator
/// with the RESHAPE now in the graph.
static int reshapeLowerToHigher(PatternRewriter &rewriter, Location loc,
                                RankedTensorType outputType, Value input1,
                                Value input2, Value &outInput1,
                                Value &outInput2) {

  int64_t input1Rank = input1.getType().cast<RankedTensorType>().getRank();
  int64_t input2Rank = input2.getType().cast<RankedTensorType>().getRank();

  Value higherTensorValue, lowerTensorValue;
  // return if rank already match
  if (input1Rank == input2Rank)
    return 1;

  if (input1Rank > input2Rank) {
    higherTensorValue = input1;
    lowerTensorValue = input2;
  } else {
    higherTensorValue = input2;
    lowerTensorValue = input1;
  }

  ArrayRef<int64_t> outputRankShape = outputType.getShape();
  ArrayRef<int64_t> higherRankShape =
      higherTensorValue.getType().cast<RankedTensorType>().getShape();
  (void)higherRankShape;
  ArrayRef<int64_t> lowerRankShape =
      lowerTensorValue.getType().cast<RankedTensorType>().getShape();

  // outputRank == higherRank == max(input1Rank, input2Rank)
  assert(higherRankShape.size() == outputRankShape.size());

  SmallVector<int64_t, 4> reshapeOutputShape;

  computeReshapeOutput(outputRankShape, lowerRankShape, reshapeOutputShape);

  auto reshapeInputType = lowerTensorValue.getType().cast<RankedTensorType>();
  auto reshapeOutputType = RankedTensorType::get(
      ArrayRef<int64_t>(reshapeOutputShape), reshapeInputType.getElementType());

  auto reshapeLower = rewriter.create<tosa::ReshapeOp>(
      loc, reshapeOutputType, lowerTensorValue,
      rewriter.getI64ArrayAttr(reshapeOutputShape));

  if (input1Rank > input2Rank) {
    outInput1 = higherTensorValue;
    outInput2 = reshapeLower.getResult();
  } else {
    outInput1 = reshapeLower.getResult();
    outInput2 = higherTensorValue;
  }

  return 0;
}

namespace {
template <typename OpTy>
struct ConvertTosaOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy tosaBinaryOp,
                                PatternRewriter &rewriter) const override {

    Value input1 = tosaBinaryOp.input1();
    Value input2 = tosaBinaryOp.input2();
    Value output = tosaBinaryOp.getResult();
    auto outputType = output.getType().cast<RankedTensorType>();

    Value outInput1, outInput2;
    if (reshapeLowerToHigher(rewriter, tosaBinaryOp.getLoc(), outputType,
                             input1, input2, outInput1, outInput2))
      return failure();

    rewriter.replaceOpWithNewOp<OpTy>(tosaBinaryOp, outputType, outInput1,
                                      outInput2);

    return success();
  }
};

// The MulOp has an extra parameter 'shift' not present in other elementwise
// binary ops, that necessitates special handling of its builder.
template <>
struct ConvertTosaOp<tosa::MulOp> : public OpRewritePattern<tosa::MulOp> {
  using OpRewritePattern<tosa::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MulOp tosaBinaryOp,
                                PatternRewriter &rewriter) const override {

    Value input1 = tosaBinaryOp.input1();
    Value input2 = tosaBinaryOp.input2();
    int32_t shift = tosaBinaryOp.shift();
    Value output = tosaBinaryOp.getResult();
    auto outputType = output.getType().cast<RankedTensorType>();

    Value outInput1, outInput2;
    if (reshapeLowerToHigher(rewriter, tosaBinaryOp.getLoc(), outputType,
                             input1, input2, outInput1, outInput2))
      return failure();

    rewriter.replaceOpWithNewOp<tosa::MulOp>(tosaBinaryOp, outputType,
                                             outInput1, outInput2, shift);

    return success();
  }
};

// The ArithmeticRightShiftOp has an extra parameter 'round' not present in
// other elementwise binary ops, that necessitates special handling of its
// builder.
template <>
struct ConvertTosaOp<tosa::ArithmeticRightShiftOp>
    : public OpRewritePattern<tosa::ArithmeticRightShiftOp> {
  using OpRewritePattern<tosa::ArithmeticRightShiftOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ArithmeticRightShiftOp tosaBinaryOp,
                                PatternRewriter &rewriter) const override {

    Value input1 = tosaBinaryOp.input1();
    Value input2 = tosaBinaryOp.input2();
    int32_t round = tosaBinaryOp.round();
    Value output = tosaBinaryOp.getResult();
    auto outputType = output.getType().dyn_cast<RankedTensorType>();

    Value outInput1, outInput2;
    if (reshapeLowerToHigher(rewriter, tosaBinaryOp.getLoc(), outputType,
                             input1, input2, outInput1, outInput2))
      return failure();

    rewriter.replaceOpWithNewOp<tosa::ArithmeticRightShiftOp>(
        tosaBinaryOp, outputType, outInput1, outInput2, round);

    return success();
  }
};
} // end anonymous namespace

namespace {
/// Pass that enables broadcast by making all input arrays have the same
/// number of dimensions. Insert RESHAPE operations to lower rank operand
struct TosaMakeBroadcastable
    : public TosaMakeBroadcastableBase<TosaMakeBroadcastable> {
public:
  void runOnFunction() override {
    auto func = getFunction();
    OwningRewritePatternList patterns(func.getContext());
    MLIRContext *ctx = func.getContext();
    // Add the generated patterns to the list.
    patterns.insert<ConvertTosaOp<tosa::AddOp>>(ctx);
    patterns.insert<ConvertTosaOp<tosa::SubOp>>(ctx);
    patterns.insert<ConvertTosaOp<tosa::MulOp>>(ctx);
    patterns.insert<ConvertTosaOp<tosa::MaximumOp>>(ctx);
    patterns.insert<ConvertTosaOp<tosa::MinimumOp>>(ctx);
    patterns.insert<ConvertTosaOp<tosa::EqualOp>>(ctx);
    patterns.insert<ConvertTosaOp<tosa::GreaterOp>>(ctx);
    patterns.insert<ConvertTosaOp<tosa::GreaterEqualOp>>(ctx);
    patterns.insert<ConvertTosaOp<tosa::LogicalLeftShiftOp>>(ctx);
    patterns.insert<ConvertTosaOp<tosa::ArithmeticRightShiftOp>>(ctx);
    patterns.insert<ConvertTosaOp<tosa::LogicalRightShiftOp>>(ctx);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::tosa::createTosaMakeBroadcastablePass() {
  return std::make_unique<TosaMakeBroadcastable>();
}
