//===- TosaCanonicalizations.cpp - Canonicalization patterns & folders ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// TOSA canonicalization patterns and folders.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tosa;

//===----------------------------------------------------------------------===//
// Operator Canonicalizers.
//===----------------------------------------------------------------------===//

struct ConcatOptimization : public OpRewritePattern<tosa::ConcatOp> {
  using OpRewritePattern<tosa::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ConcatOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInput1().size() != 1)
      return failure();
    if (op.getInput1().front().getType() != op.getType()) {
      rewriter
          .replaceOpWithNewOp<tensor::CastOp>(op, op.getType(),
                                              op.getInput1().front())
          .getResult();
      return success();
    }

    rewriter.replaceOp(op, op.getInput1().front());
    return success();
  }
};

void ConcatOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<ConcatOptimization>(context);
}

struct ReshapeReshapeOptimization : public OpRewritePattern<tosa::ReshapeOp> {
  using OpRewritePattern<tosa::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput1();
    Operation *definingOp = input.getDefiningOp();
    if (!definingOp)
      return failure();

    if (tosa::ReshapeOp reshapeOp = dyn_cast<tosa::ReshapeOp>(definingOp)) {
      rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
          op, op.getType(), reshapeOp.getInput1(), op.getNewShape());
      return success();
    }

    return failure();
  }
};

struct ReshapeConstOptimization : public OpRewritePattern<tosa::ReshapeOp> {
  using OpRewritePattern<tosa::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput1();
    ArrayAttr newShape = op.getNewShape();

    // Check if input is constant
    DenseElementsAttr inputAttr;
    if (!matchPattern(input, m_Constant(&inputAttr)))
      return failure();

    // Check if has >1 consumer and is not splat
    if (!input.hasOneUse() && !inputAttr.isSplat())
      return failure();

    // Grab the new shape
    SmallVector<int64_t> newShapeValues = llvm::to_vector<6>(
        llvm::map_range(newShape.getValue(), [](const Attribute &val) {
          return val.cast<IntegerAttr>().getValue().getSExtValue();
        }));

    // Build new const op with correct output shape
    ShapedType inputShape = input.getType().cast<ShapedType>();
    DenseElementsAttr outputAttr =
        inputAttr.reshape(inputShape.clone(newShapeValues));
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, outputAttr.getType(),
                                               outputAttr);
    return success();
  }
};

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptimization>(context);
  results.add<ReshapeConstOptimization>(context);
}

LogicalResult SelectOp::canonicalize(SelectOp op, PatternRewriter &rewriter) {
  auto notOp = op.getPred().getDefiningOp<tosa::LogicalNotOp>();
  if (!notOp)
    return failure();
  rewriter.updateRootInPlace(op, [&]() {
    op.getOperation()->setOperands(
        {notOp.getInput1(), op.getOnFalse(), op.getOnTrue()});
  });
  return success();
}

struct NoOpOptimization : public OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto perm = op.getPerms();

    DenseIntElementsAttr permAttr;
    if (!matchPattern(perm, m_Constant(&permAttr))) {
      return failure();
    }

    SmallVector<int64_t> permValues = llvm::to_vector<6>(
        llvm::map_range(permAttr.getValues<APInt>(),
                        [](const APInt &val) { return val.getSExtValue(); }));

    for (int i = 0, s = permValues.size(); i < s; i++) {
      if (i != permValues[i]) {
        return failure();
      }
    }

    rewriter.replaceOp(op, op.getInput1());
    return success();
  }
};

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<NoOpOptimization>(context);
}

struct AddZeroOptimization : public OpRewritePattern<tosa::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto input1 = op.getInput1();
    auto input2 = op.getInput2();

    DenseElementsAttr input1Attr;
    if (matchPattern(input1, m_Constant(&input1Attr)) && input1Attr.isSplat() &&
        input2.getType() == op.getType()) {
      if (input1Attr.getType().getElementType().isa<IntegerType>() &&
          input1Attr.getSplatValue<APInt>().isZero()) {
        rewriter.replaceOp(op, op.getInput2());
        return success();
      }
    }

    DenseElementsAttr input2Attr;
    if (matchPattern(input2, m_Constant(&input2Attr)) && input2Attr.isSplat() &&
        input1.getType() == op.getType()) {
      if (input2Attr.getType().getElementType().isa<IntegerType>() &&
          input2Attr.getSplatValue<APInt>().isZero()) {
        rewriter.replaceOp(op, op.getInput1());
        return success();
      }
    }

    return failure();
  }
};

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<AddZeroOptimization>(context);
}

struct MulOneOptimization : public OpRewritePattern<tosa::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MulOp op,
                                PatternRewriter &rewriter) const override {
    auto input1 = op.getInput1();
    auto input2 = op.getInput2();

    DenseElementsAttr input1Attr;
    if (matchPattern(input1, m_Constant(&input1Attr)) && input1Attr.isSplat() &&
        input2.getType() == op.getType()) {
      if (input1Attr.getType().getElementType().isa<FloatType>() &&
          input1Attr.getSplatValue<APFloat>().isExactlyValue(1)) {
        rewriter.replaceOp(op, op.getInput2());
        return success();
      }

      if (input1Attr.getType().getElementType().isa<IntegerType>() &&
          matchPattern(input1, m_One())) {
        rewriter.replaceOp(op, op.getInput2());
        return success();
      }
    }

    DenseElementsAttr input2Attr;
    if (matchPattern(input2, m_Constant(&input2Attr)) && input2Attr.isSplat() &&
        input1.getType() == op.getType()) {
      if (input2Attr.getType().getElementType().isa<FloatType>() &&
          input2Attr.getSplatValue<APFloat>().isExactlyValue(1)) {
        rewriter.replaceOp(op, op.getInput1());
        return success();
      }

      if (input2Attr.getType().getElementType().isa<IntegerType>() &&
          matchPattern(input2, m_One())) {
        rewriter.replaceOp(op, op.getInput1());
        return success();
      }
    }

    return failure();
  }
};

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<MulOneOptimization>(context);
}

struct MaterializePadValue : public OpRewritePattern<tosa::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::PadOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getPadConst())
      return failure();

    auto input = op.getInput1();
    auto padding = op.getPadding();

    ShapedType inputTy = input.getType().cast<ShapedType>();
    Type elementTy = inputTy.getElementType();

    Attribute constantAttr;
    if (elementTy.isa<FloatType>()) {
      constantAttr = rewriter.getFloatAttr(elementTy, 0.0);
    } else if (elementTy.isa<IntegerType>() && !op.getQuantizationInfo()) {
      constantAttr = rewriter.getIntegerAttr(elementTy, 0);
    } else if (elementTy.isa<IntegerType>() && op.getQuantizationInfo()) {
      auto value = op.getQuantizationInfo()->getInputZp();
      constantAttr = rewriter.getIntegerAttr(elementTy, value);
    }

    if (!constantAttr) {
      return rewriter.notifyMatchFailure(
          op,
          "tosa.pad to linalg lowering encountered an unknown element type");
    }

    auto denseAttr = DenseElementsAttr::get(
        RankedTensorType::get({}, elementTy), constantAttr);
    auto constantVal = rewriter.create<tosa::ConstOp>(
        op.getLoc(), denseAttr.getType(), denseAttr);

    rewriter.replaceOpWithNewOp<tosa::PadOp>(
        op, op.getType(), ValueRange{input, padding, constantVal},
        op->getAttrs());
    return success();
  }
};

void PadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<MaterializePadValue>(context);
}

struct MaxPool2dIsNoOp : public OpRewritePattern<tosa::MaxPool2dOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MaxPool2dOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    Value output = op.getOutput();
    ShapedType inputType = input.getType().cast<ShapedType>();
    ShapedType outputType = output.getType().cast<ShapedType>();

    if (!inputType.hasStaticShape() || !outputType.hasStaticShape()) {
      return failure();
    }

    // If the output and input shapes are 1x1, then this is a no op.
    ArrayRef<int64_t> outputShape = outputType.getShape();
    if (outputShape[1] != 1 || outputShape[2] != 1) {
      return failure();
    }

    ArrayRef<int64_t> inputShape = inputType.getShape();
    if (inputShape[1] != 1 || inputShape[2] != 1) {
      return failure();
    }

    rewriter.replaceOp(op, input);
    return success();
  }
};

void MaxPool2dOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<MaxPool2dIsNoOp>(context);
}

struct ClampIsNoOp : public OpRewritePattern<tosa::ClampOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ClampOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    auto inputType =
        op.getInput().getType().template dyn_cast<RankedTensorType>();
    auto inputElementType = inputType.getElementType();

    if (!inputType.hasStaticShape()) {
      return failure();
    }

    if (inputElementType.isF32()) {
      auto minClamp = op.getMinFp();
      auto maxClamp = op.getMaxFp();
      bool isMin = (minClamp.isLargest() || minClamp.isInfinity()) &&
                   minClamp.isNegative();
      bool isMax = (maxClamp.isLargest() || maxClamp.isInfinity()) &&
                   !maxClamp.isNegative();

      if (isMin && isMax) {
        rewriter.replaceOp(op, input);
        return success();
      }
      return failure();
    }

    if (inputElementType.isUnsignedInteger()) {
      int64_t minClamp = op.getMinInt();
      int64_t maxClamp = op.getMaxInt();

      int64_t intMin =
          APInt::getMinValue(inputElementType.getIntOrFloatBitWidth())
              .getZExtValue();
      int64_t intMax =
          APInt::getMaxValue(inputElementType.getIntOrFloatBitWidth())
              .getZExtValue();

      if (minClamp <= intMin && maxClamp >= intMax) {
        rewriter.replaceOp(op, input);
        return success();
      }
      return failure();
    }

    if (inputElementType.isa<IntegerType>()) {
      int64_t minClamp = op.getMinInt();
      int64_t maxClamp = op.getMaxInt();

      int64_t intMin =
          APInt::getSignedMinValue(inputElementType.getIntOrFloatBitWidth())
              .getSExtValue();
      int64_t intMax =
          APInt::getSignedMaxValue(inputElementType.getIntOrFloatBitWidth())
              .getSExtValue();

      if (minClamp <= intMin && maxClamp >= intMax) {
        rewriter.replaceOp(op, input);
        return success();
      }
      return failure();
    }

    return failure();
  }
};

struct ClampClampOptimization : public OpRewritePattern<tosa::ClampOp> {
  using OpRewritePattern<tosa::ClampOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ClampOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getInput();

    Operation *definingOp = input.getDefiningOp();
    if (!definingOp)
      return failure();

    if (tosa::ClampOp clampOp = dyn_cast<tosa::ClampOp>(definingOp)) {
      auto minFp = std::max(op.getMinFp(), clampOp.getMinFp()).convertToFloat();
      auto maxFp = std::min(op.getMaxFp(), clampOp.getMaxFp()).convertToFloat();

      auto minInt = std::max(op.getMinInt(), clampOp.getMinInt());
      auto maxInt = std::min(op.getMaxInt(), clampOp.getMaxInt());

      rewriter.replaceOpWithNewOp<tosa::ClampOp>(
          op, op.getType(), clampOp.getInput(),
          rewriter.getI64IntegerAttr(minInt),
          rewriter.getI64IntegerAttr(maxInt), rewriter.getF32FloatAttr(minFp),
          rewriter.getF32FloatAttr(maxFp));
      return success();
    }

    return failure();
  }
};

void ClampOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<ClampIsNoOp>(context);
  results.add<ClampClampOptimization>(context);
}

//===----------------------------------------------------------------------===//
// Operator Folders.
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(ArrayRef<Attribute> operands) {
  if (getInput().getType() == getType())
    return getInput();
  return {};
}

OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValueAttr();
}

#define REDUCE_FOLDER(OP)                                                      \
  OpFoldResult OP::fold(ArrayRef<Attribute> operands) {                        \
    ShapedType inputTy = input().getType().cast<ShapedType>();                 \
    if (!inputTy.hasRank())                                                    \
      return {};                                                               \
    if (inputTy.getDimSize(axis()) == 1)                                       \
      return input();                                                          \
    return {};                                                                 \
  }

REDUCE_FOLDER(ReduceAllOp)
REDUCE_FOLDER(ReduceAnyOp)
REDUCE_FOLDER(ReduceMaxOp)
REDUCE_FOLDER(ReduceMinOp)
REDUCE_FOLDER(ReduceProdOp)
REDUCE_FOLDER(ReduceSumOp)
#undef REDUCE_FOLDER

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  auto inputTy = getInput1().getType().dyn_cast<RankedTensorType>();
  auto outputTy = getType().dyn_cast<RankedTensorType>();

  if (!inputTy || !outputTy || inputTy != outputTy)
    return {};
  return getInput1();
}

OpFoldResult PadOp::fold(ArrayRef<Attribute> operands) {
  // If the pad is all zeros we can fold this operation away.
  if (operands[1]) {
    auto densePad = operands[1].cast<DenseElementsAttr>();
    if (densePad.isSplat() && densePad.getSplatValue<APInt>().isZero()) {
      return getInput1();
    }
  }

  return {};
}

OpFoldResult SliceOp::fold(ArrayRef<Attribute> operands) {
  auto inputTy = getInput().getType().dyn_cast<RankedTensorType>();
  auto outputTy = getType().dyn_cast<RankedTensorType>();

  if (!inputTy || !outputTy || inputTy != outputTy)
    return {};
  if (inputTy.hasStaticShape())
    return getInput();

  return {};
}

OpFoldResult tosa::SelectOp::fold(ArrayRef<Attribute> operands) {
  if (getOnTrue() == getOnFalse())
    return getOnTrue();

  auto predicate = operands[0].dyn_cast_or_null<DenseIntElementsAttr>();
  if (!predicate)
    return {};

  if (!predicate.isSplat())
    return {};
  return predicate.getSplatValue<APInt>().getBoolValue() ? getOnTrue()
                                                         : getOnFalse();
}

OpFoldResult TileOp::fold(ArrayRef<Attribute> operands) {
  bool allOnes = true;
  for (Attribute val : getMultiples().getValue()) {
    allOnes = allOnes && val.cast<IntegerAttr>().getValue().getSExtValue() == 1;
  }

  if (allOnes && getInput1().getType() == getType())
    return getInput1();
  return {};
}

OpFoldResult TransposeOp::fold(ArrayRef<Attribute> operands) {
  if (!operands[1])
    return {};

  // Transposing splat values just means reshaping.
  if (auto input = operands[0].dyn_cast_or_null<DenseElementsAttr>()) {
    if (input.isSplat())
      return input.reshape(getType().cast<ShapedType>());
  }

  auto perms = llvm::to_vector<6>(llvm::map_range(
      operands[1].cast<DenseIntElementsAttr>().getValues<APInt>(),
      [](const APInt &val) { return val.getSExtValue(); }));

  if (llvm::equal(llvm::seq<int64_t>(0, perms.size()), perms) &&
      getInput1().getType() == getType())
    return getInput1();
  return {};
}
