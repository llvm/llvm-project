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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAMAKEBROADCASTABLEPASS
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

/// There are two potential ways implementing broadcast:
/// a. https://www.tensorflow.org/xla/broadcasting#formal_definition
/// b. https://numpy.org/doc/stable/user/basics.broadcasting.html
/// This pass implements b (numpy style) now.

/// In this pass, we insert RESHAPE operators to increase the rank of the
/// lower rank operand as a first step in the broadcasting process. The TOSA
/// operators that support broadcast require that the rank of the operands
/// are equal.

// Examples:
// If lower=[c], higher=[a, b, c], [c] reshaped into [1, 1, c].
// If lower=[b, c], higher=[a, b, c], [b, c] reshaped into [1, b, c].
// If lower=[a], higher=[a, a], [a] reshaped into [1, a].
// If lower=[a], target=[a, b, a], [a] reshaped into [1, 1, a].
// If lower=[], target=[a, b, c], [] reshaped into [1, 1, 1].

static LogicalResult
computeReshapeOutput(ArrayRef<int64_t> higherRankShape,
                     ArrayRef<int64_t> lowerRankShape,
                     SmallVectorImpl<int64_t> &reshapeOutputShape) {
  // Initialize new shapes with [1] * higherRank.
  int64_t higherRank = higherRankShape.size();
  int64_t lowerRank = lowerRankShape.size();

  reshapeOutputShape.assign(higherRank, 1);

  int64_t higherRankDim;
  int64_t lowerRankDim;

  for (int64_t i = higherRank - 1, j = lowerRank - 1; i >= 0 && j >= 0;
       i--, j--) {
    higherRankDim = higherRankShape[i];
    lowerRankDim = lowerRankShape[j];

    if (lowerRankDim == 1 && higherRankDim > 1)
      reshapeOutputShape[i] = 1;
    else if ((lowerRankDim > 1 && higherRankDim == 1) ||
             (lowerRankDim == higherRankDim))
      reshapeOutputShape[i] = lowerRankDim;
    else if (higherRankDim != lowerRankDim)
      return failure();
  }
  return success();
}

/// Common code to create the reshape op where necessary to make the rank of the
/// operations equal. Returns the updated input1 and input2 for the original
/// input. The caller is expected to use these to rewrite the original operator
/// with the RESHAPE now in the graph.
static LogicalResult reshapeLowerToHigher(PatternRewriter &rewriter,
                                          Location loc,
                                          RankedTensorType outputType,
                                          Value input1, Value input2,
                                          Value &outInput1, Value &outInput2) {
  auto input1Ty = input1.getType().dyn_cast<RankedTensorType>();
  auto input2Ty = input2.getType().dyn_cast<RankedTensorType>();

  if (!input1Ty || !input2Ty)
    return failure();

  int64_t input1Rank = input1Ty.getRank();
  int64_t input2Rank = input2Ty.getRank();

  Value higherTensorValue, lowerTensorValue;
  // Cannot rewrite as its already correct.
  if (input1Rank == input2Rank)
    return failure();

  if (input1Rank > input2Rank) {
    higherTensorValue = input1;
    lowerTensorValue = input2;
  } else {
    higherTensorValue = input2;
    lowerTensorValue = input1;
  }

  ArrayRef<int64_t> higherRankShape =
      higherTensorValue.getType().cast<RankedTensorType>().getShape();
  (void)higherRankShape;
  ArrayRef<int64_t> lowerRankShape =
      lowerTensorValue.getType().cast<RankedTensorType>().getShape();

  SmallVector<int64_t, 4> reshapeOutputShape;

  if (computeReshapeOutput(higherRankShape, lowerRankShape, reshapeOutputShape)
          .failed())
    return failure();

  auto reshapeInputType = lowerTensorValue.getType().cast<RankedTensorType>();
  auto reshapeOutputType = RankedTensorType::get(
      ArrayRef<int64_t>(reshapeOutputShape), reshapeInputType.getElementType());

  // Verify the rank agrees with the output type if the output type is ranked.
  if (outputType) {
    if (outputType.getShape().size() != reshapeOutputShape.size() ||
        outputType.getShape().size() != higherRankShape.size())
      return failure();
  }

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

  return success();
}

namespace {
template <typename OpTy>
struct ConvertTosaOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy tosaBinaryOp,
                                PatternRewriter &rewriter) const override {

    Value input1 = tosaBinaryOp.getInput1();
    Value input2 = tosaBinaryOp.getInput2();
    Value output = tosaBinaryOp.getResult();

    auto outputType = output.getType().dyn_cast<RankedTensorType>();
    if (!outputType)
      return failure();

    Value outInput1, outInput2;
    if (reshapeLowerToHigher(rewriter, tosaBinaryOp.getLoc(), outputType,
                             input1, input2, outInput1, outInput2)
            .failed())
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

    Value input1 = tosaBinaryOp.getInput1();
    Value input2 = tosaBinaryOp.getInput2();
    int32_t shift = tosaBinaryOp.getShift();
    Value output = tosaBinaryOp.getResult();
    auto outputType = output.getType().dyn_cast<RankedTensorType>();
    if (!outputType)
      return failure();

    Value outInput1, outInput2;
    if (reshapeLowerToHigher(rewriter, tosaBinaryOp.getLoc(), outputType,
                             input1, input2, outInput1, outInput2)
            .failed())
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

    Value input1 = tosaBinaryOp.getInput1();
    Value input2 = tosaBinaryOp.getInput2();
    int32_t round = tosaBinaryOp.getRound();
    Value output = tosaBinaryOp.getResult();
    auto outputType = output.getType().dyn_cast<RankedTensorType>();
    if (!outputType)
      return failure();

    Value outInput1, outInput2;
    if (reshapeLowerToHigher(rewriter, tosaBinaryOp.getLoc(), outputType,
                             input1, input2, outInput1, outInput2)
            .failed())
      return failure();

    rewriter.replaceOpWithNewOp<tosa::ArithmeticRightShiftOp>(
        tosaBinaryOp, outputType, outInput1, outInput2, round);

    return success();
  }
};
} // namespace

namespace {
/// Pass that enables broadcast by making all input arrays have the same
/// number of dimensions. Insert RESHAPE operations to lower rank operand
struct TosaMakeBroadcastablePass
    : public tosa::impl::TosaMakeBroadcastablePassBase<
          TosaMakeBroadcastablePass> {
public:
  using TosaMakeBroadcastablePassBase::TosaMakeBroadcastablePassBase;

  void runOnOperation() override {
    auto func = getOperation();
    RewritePatternSet patterns(func.getContext());
    MLIRContext *ctx = func.getContext();
    // Add the generated patterns to the list.
    patterns.add<ConvertTosaOp<tosa::BitwiseAndOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::BitwiseOrOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::BitwiseXorOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::AddOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::SubOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::MulOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::DivOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::MaximumOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::MinimumOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::EqualOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::GreaterOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::GreaterEqualOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::LogicalLeftShiftOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::ArithmeticRightShiftOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::LogicalRightShiftOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::LogicalAndOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::LogicalOrOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::LogicalXorOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::PowOp>>(ctx);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace
