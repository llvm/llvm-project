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
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAMAKEBROADCASTABLE
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

/// Common code to create the reshape op where necessary to make the rank of the
/// operations equal. input1 and input2 will be updated when the rank has
/// changed. The caller is expected to use these to rewrite the original
/// operator with the RESHAPE now in the graph.
/// return failure when (1) no reshape needed, or (2) output_type is specified
/// and it has different rank
LogicalResult reshapeLowerToHigher(PatternRewriter &rewriter, Location loc,
                                   RankedTensorType outputType, Value &input1,
                                   Value &input2) {
  auto input1Ty = dyn_cast<RankedTensorType>(input1.getType());
  auto input2Ty = dyn_cast<RankedTensorType>(input2.getType());

  if (!input1Ty || !input2Ty) {
    return rewriter.notifyMatchFailure(loc, "input not a ranked tensor");
  }

  int64_t input1Rank = input1Ty.getRank();
  int64_t input2Rank = input2Ty.getRank();

  if (input1Rank == input2Rank)
    return rewriter.notifyMatchFailure(loc,
                                       "cannot rewrite as its already correct");

  Value input1Copy = input1;
  Value input2Copy = input2;
  if (EqualizeRanks(rewriter, loc, input1Copy, input2Copy).failed()) {
    return rewriter.notifyMatchFailure(loc, "failed to reshape inputs");
  }

  // Verify the rank agrees with the output type if the output type is ranked.
  if (outputType) {
    if (outputType.getRank() !=
            llvm::cast<RankedTensorType>(input1Copy.getType()).getRank() ||
        outputType.getRank() !=
            llvm::cast<RankedTensorType>(input2Copy.getType()).getRank())
      return rewriter.notifyMatchFailure(
          loc, "the reshaped type doesn't agrees with the ranked output type");
  }

  input1 = input1Copy;
  input2 = input2Copy;

  return success();
}

template <typename OpTy>
struct ConvertTosaOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy tosaBinaryOp,
                                PatternRewriter &rewriter) const override {

    Value input1 = tosaBinaryOp.getInput1();
    Value input2 = tosaBinaryOp.getInput2();
    Value output = tosaBinaryOp.getResult();

    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType)
      return failure();

    if (reshapeLowerToHigher(rewriter, tosaBinaryOp.getLoc(), outputType,
                             input1, input2)
            .failed())
      return failure();

    rewriter.replaceOpWithNewOp<OpTy>(tosaBinaryOp, outputType, input1, input2);

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
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType)
      return failure();

    if (reshapeLowerToHigher(rewriter, tosaBinaryOp.getLoc(), outputType,
                             input1, input2)
            .failed())
      return failure();

    rewriter.replaceOpWithNewOp<tosa::MulOp>(tosaBinaryOp, outputType, input1,
                                             input2, shift);

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
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType)
      return failure();

    if (reshapeLowerToHigher(rewriter, tosaBinaryOp.getLoc(), outputType,
                             input1, input2)
            .failed())
      return failure();

    rewriter.replaceOpWithNewOp<tosa::ArithmeticRightShiftOp>(
        tosaBinaryOp, outputType, input1, input2, round);

    return success();
  }
};

template <>
struct ConvertTosaOp<tosa::SelectOp> : public OpRewritePattern<tosa::SelectOp> {
  using OpRewritePattern<tosa::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::SelectOp tosaOp,
                                PatternRewriter &rewriter) const override {

    Value input1 = tosaOp.getPred();
    Value input2 = tosaOp.getOnTrue();
    Value input3 = tosaOp.getOnFalse();
    Value output = tosaOp.getResult();

    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(tosaOp, "output not a ranked tensor");

    // Apply broadcasting to each pair of inputs separately, and chain them as
    // compound as below so that the broadcasting happens all at once.
    bool reshaped1 = reshapeLowerToHigher(rewriter, tosaOp.getLoc(), outputType,
                                          input1, input2)
                         .succeeded();

    bool reshaped2 = reshapeLowerToHigher(rewriter, tosaOp.getLoc(), outputType,
                                          input1, input3)
                         .succeeded();

    bool reshaped3 = reshapeLowerToHigher(rewriter, tosaOp.getLoc(), outputType,
                                          input2, input3)
                         .succeeded();

    if (!reshaped1 && !reshaped2 && !reshaped3)
      return rewriter.notifyMatchFailure(
          tosaOp,
          "cannot rewrite as the rank of all operands is already aligned");

    int32_t result1Rank = cast<RankedTensorType>(input1.getType()).getRank();
    int32_t result2Rank = cast<RankedTensorType>(input2.getType()).getRank();
    int32_t result3Rank = cast<RankedTensorType>(input3.getType()).getRank();
    int32_t outputRank = outputType.getRank();

    if ((result1Rank != result2Rank) || (result2Rank != result3Rank) ||
        (result1Rank != outputRank))
      return rewriter.notifyMatchFailure(
          tosaOp, "not all ranks are aligned with each other");

    rewriter.replaceOpWithNewOp<tosa::SelectOp>(tosaOp, outputType, input1,
                                                input2, input3);

    return success();
  }
};
} // namespace

namespace {
/// Pass that enables broadcast by making all input arrays have the same
/// number of dimensions. Insert RESHAPE operations to lower rank operand
struct TosaMakeBroadcastable
    : public tosa::impl::TosaMakeBroadcastableBase<TosaMakeBroadcastable> {
public:
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
    patterns.add<ConvertTosaOp<tosa::IntDivOp>>(ctx);
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
    patterns.add<ConvertTosaOp<tosa::SelectOp>>(ctx);
    patterns.add<ConvertTosaOp<tosa::PowOp>>(ctx);
    (void)applyPatternsGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaMakeBroadcastablePass() {
  return std::make_unique<TosaMakeBroadcastable>();
}
