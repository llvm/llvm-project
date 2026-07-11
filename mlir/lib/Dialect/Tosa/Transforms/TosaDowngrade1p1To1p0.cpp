//===- TosaDowngrade1_1To1_0.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites constructs which are only compatible in TOSA specification 1.1 and
// above to their TOSA 1.0 counterparts where possible. Downgrading is
// best-effort and validation should be performed afterwards to ensure
// compatibility with the TOSA 1.0 specification.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSADOWNGRADE1P1TO1P0PASS
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

class BoolFp32CastRewrite : public OpRewritePattern<tosa::CastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::CastOp op,
                                PatternRewriter &rewriter) const override {
    const Value input = op.getInput();

    const Type i1Type = rewriter.getI1Type();
    const Type f32Type = rewriter.getF32Type();

    const Type inputElemType = getElementTypeOrSelf(input.getType());
    const Type outputElemType = getElementTypeOrSelf(op.getType());
    const bool isFp32ToBool =
        inputElemType == f32Type && outputElemType == i1Type;
    const bool isBoolToFp32 =
        inputElemType == i1Type && outputElemType == f32Type;

    if (!isFp32ToBool && !isBoolToFp32)
      return rewriter.notifyMatchFailure(op,
                                         "expected cast between bool and f32");

    const Type outputType = op.getType();
    const Type i8Type = rewriter.getI8Type();
    const Type intermediateType = cast<TensorType>(outputType).clone(i8Type);

    auto inner =
        tosa::CastOp::create(rewriter, op.getLoc(), intermediateType, input);
    auto outer = tosa::CastOp::create(rewriter, op.getLoc(), outputType,
                                      inner.getOutput());
    rewriter.replaceOp(op, outer.getOutput());
    return success();
  }
};

class BoolGatherRewrite : public OpRewritePattern<tosa::GatherOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::GatherOp op,
                                PatternRewriter &rewriter) const override {
    const Value values = op.getValues();
    const Value indices = op.getIndices();

    const Type valuesType = values.getType();
    const Type resultType = op.getType();

    const Type i1Type = rewriter.getI1Type();
    const Type i32Type = rewriter.getI32Type();
    if (getElementTypeOrSelf(valuesType) != i1Type ||
        getElementTypeOrSelf(indices.getType()) != i32Type)
      return rewriter.notifyMatchFailure(
          op, "expected values of bool type and indices of i32 type");

    const Type i8Type = rewriter.getI8Type();
    const Type valuesI8Type = cast<TensorType>(valuesType).clone(i8Type);
    const Type resultI8Type = cast<TensorType>(resultType).clone(i8Type);

    auto valuesToI8 =
        tosa::CastOp::create(rewriter, op.getLoc(), valuesI8Type, values);
    auto gatherI8 = tosa::GatherOp::create(rewriter, op.getLoc(), resultI8Type,
                                           valuesToI8.getOutput(), indices);
    auto i8ToBool = tosa::CastOp::create(rewriter, op.getLoc(), resultType,
                                         gatherI8.getOutput());
    rewriter.replaceOp(op, i8ToBool.getOutput());
    return success();
  }
};

class BoolScatterRewrite : public OpRewritePattern<tosa::ScatterOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    const Value valuesIn = op.getValuesIn();
    const Value indices = op.getIndices();

    const Type valuesInType = valuesIn.getType();
    const Type i1Type = rewriter.getI1Type();
    const Type i32Type = rewriter.getI32Type();
    if (getElementTypeOrSelf(valuesInType) != i1Type ||
        getElementTypeOrSelf(indices.getType()) != i32Type)
      return rewriter.notifyMatchFailure(
          op, "expected values of bool type and indices of i32 type");

    const Value input = op.getInput();
    const Type inputType = input.getType();
    const Type resultType = op.getType();

    const Type i8Type = rewriter.getI8Type();
    const Type valuesInI8Type = cast<TensorType>(valuesInType).clone(i8Type);
    const Type inputI8Type = cast<TensorType>(inputType).clone(i8Type);
    const Type resultI8Type = cast<TensorType>(resultType).clone(i8Type);

    auto valuesInToI8 =
        tosa::CastOp::create(rewriter, op.getLoc(), valuesInI8Type, valuesIn);
    auto inputToI8 =
        tosa::CastOp::create(rewriter, op.getLoc(), inputI8Type, input);
    auto scatterI8 = tosa::ScatterOp::create(
        rewriter, op.getLoc(), resultI8Type, valuesInToI8.getOutput(), indices,
        inputToI8.getOutput());
    auto i8ToBool = tosa::CastOp::create(rewriter, op.getLoc(), resultType,
                                         scatterI8.getValuesOut());
    rewriter.replaceOp(op, i8ToBool.getOutput());
    return success();
  }
};

static LogicalResult isMatMulTTypeCompatibleForDowngrade(tosa::MatMulTOp op) {
  const Type aElementType = getStorageElementTypeOrSelf(op.getA().getType());
  const Type bElementType = getStorageElementTypeOrSelf(op.getB().getType());
  const Type outputElementType =
      getStorageElementTypeOrSelf(op.getOutput().getType());

  if (aElementType != bElementType)
    return failure();

  if ((aElementType.isF16() && outputElementType.isF16()) ||
      (aElementType.isF16() && outputElementType.isF32()) ||
      (aElementType.isF32() && outputElementType.isF32()) ||
      (aElementType.isBF16() && outputElementType.isF32()) ||
      (aElementType.isInteger(8) && outputElementType.isInteger(32)) ||
      (aElementType.isInteger(16) && outputElementType.isInteger(48)) ||
      (isa<Float8E5M2Type>(aElementType) && outputElementType.isF16()) ||
      (isa<Float8E4M3FNType>(aElementType) && outputElementType.isF16()))
    return success();

  return failure();
}

class MatMulTRewrite : public OpRewritePattern<tosa::MatMulTOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MatMulTOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(isMatMulTTypeCompatibleForDowngrade(op)))
      return rewriter.notifyMatchFailure(
          op, "expected 1.0-compatible matmul_t element types");

    const Type aType = op.getA().getType();
    const Type bType = op.getB().getType();
    const ShapeAdaptor aShape(aType);
    const ShapeAdaptor bShape(bType);
    if (!aShape.hasRank() || !bShape.hasRank())
      return rewriter.notifyMatchFailure(op, "expected ranked A and B tensors");

    const int64_t dSize = bShape.getDimSize(0);
    const int64_t nSize = aShape.getDimSize(0);

    // To convert broadcasting behaviour to TOSA 1.0, we're required to tile the
    // input. TOSA 1.0 does not support shape expressions, so the batch size
    // must be known at compile time.
    if (ShapedType::isDynamic(dSize) ||
        (dSize == 1 && ShapedType::isDynamic(nSize)))
      return rewriter.notifyMatchFailure(
          op, "expected known batch size for broadcast");

    const int64_t wSize = bShape.getDimSize(1);
    const int64_t cSize = bShape.getDimSize(2);
    const Location loc = op.getLoc();
    const RankedTensorType transposedBType =
        cast<RankedTensorType>(bType).clone({dSize, cSize, wSize});
    auto transpose =
        tosa::TransposeOp::create(rewriter, loc, transposedBType, op.getB(),
                                  rewriter.getDenseI32ArrayAttr({0, 2, 1}));
    Value matMulB = transpose.getOutput();

    // Matmul does not support broadcasting, so tile b if required
    if (dSize == 1 && nSize != 1) {
      const RankedTensorType tiledBType =
          cast<RankedTensorType>(bType).clone({nSize, cSize, wSize});
      const Value multiples = getTosaConstShape(rewriter, loc, {nSize, 1, 1});
      auto tile =
          tosa::TileOp::create(rewriter, loc, tiledBType, matMulB, multiples);
      matMulB = tile.getOutput();
    }

    auto matmul = tosa::MatMulOp::create(rewriter, loc, op.getType(), op.getA(),
                                         matMulB, op.getAZp(), op.getBZp());
    rewriter.replaceOp(op, matmul.getOutput());
    return success();
  }
};

struct TosaDowngrade1p1To1p0Pass
    : public tosa::impl::TosaDowngrade1p1To1p0PassBase<
          TosaDowngrade1p1To1p0Pass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext &context = getContext();
    func::FuncOp func = getOperation();

    RewritePatternSet patterns(&context);
    patterns.add<BoolFp32CastRewrite, BoolGatherRewrite, BoolScatterRewrite,
                 MatMulTRewrite>(&context);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    if (failed(applyPatternsGreedily(func, frozenPatterns)))
      return signalPassFailure();
  }
};

} // namespace
