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
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include <functional>

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
    ShapedType inputTy = input.getType().cast<ShapedType>();
    ShapedType resultTy = op.getType().cast<ShapedType>();

    if (inputTy.getElementType() != resultTy.getElementType())
      return rewriter.notifyMatchFailure(op, "element type does not match.");

    // Check if input is constant
    DenseElementsAttr inputAttr;
    if (!matchPattern(input, m_Constant(&inputAttr)))
      return rewriter.notifyMatchFailure(op, "Non-constant input.");

    // Check if has >1 consumer and is not splat
    if (!input.hasOneUse() && !inputAttr.isSplat())
      return rewriter.notifyMatchFailure(op,
                                         "Used more than once or not-splat");

    // Build new const op with correct output shape
    ShapedType inputShape = input.getType().cast<ShapedType>();
    DenseElementsAttr outputAttr =
        inputAttr.reshape(inputShape.clone(op.getNewShape()));
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

struct ConsolidateTransposeOptimization
    : public OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    // Input is also TransposeOp - transpose(transpose(A)).
    auto innerTranspose =
        transposeOp.getInput1().getDefiningOp<tosa::TransposeOp>();
    if (!innerTranspose)
      return rewriter.notifyMatchFailure(transposeOp,
                                         "input must be transpose operation");

    SmallVector<int64_t> transposePerms, innerTransposePerms;
    if (transposeOp.getConstantPerms(transposePerms).failed())
      return rewriter.notifyMatchFailure(transposeOp,
                                         "transpose perms must be constant");
    if (innerTranspose.getConstantPerms(innerTransposePerms).failed())
      return rewriter.notifyMatchFailure(
          transposeOp, "inner transpose perms must be constant");
    if (transposePerms.size() != innerTransposePerms.size())
      return rewriter.notifyMatchFailure(
          transposeOp,
          "transpose and inner transpose perms sizes must be equal");
    if (transposePerms.empty())
      return rewriter.notifyMatchFailure(
          transposeOp, "transpose perms sizes must be positive");

    // Consolidate transposes into one transpose.
    SmallVector<int32_t> perms(transposePerms.size());
    for (int i = 0, s = transposePerms.size(); i < s; ++i)
      perms[i] = innerTransposePerms[transposePerms[i]];

    auto permsTy =
        RankedTensorType::get(transposePerms.size(), rewriter.getI32Type());
    auto permsAttr = DenseIntElementsAttr::get(permsTy, perms);
    Value permsValue =
        rewriter.create<arith::ConstantOp>(transposeOp.getLoc(), permsAttr);

    rewriter.replaceOpWithNewOp<tosa::TransposeOp>(
        transposeOp, transposeOp.getResult().getType(),
        innerTranspose.getInput1(), permsValue);

    return success();
  }
};

// Determines the case when tosa.transpose is a tosa.reshape operation.
struct TransposeIsReshape : public OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    DenseIntElementsAttr permAttr;
    if (!matchPattern(op.getPerms(), m_Constant(&permAttr)))
      return rewriter.notifyMatchFailure(op, "Non-constant permutation");

    auto input = op.getInput1();
    auto inputTy = input.getType().cast<ShapedType>();
    if (!inputTy.hasRank())
      return rewriter.notifyMatchFailure(op, "Unranked input.");

    int64_t numDynDims = 0;
    for (int i = 0; i < inputTy.getRank(); ++i)
      if (inputTy.isDynamicDim(i))
        numDynDims++;

    if (numDynDims > 1)
      return rewriter.notifyMatchFailure(op, "Has more than one dynamic dim.");

    SmallVector<int64_t> permValues = llvm::to_vector<6>(
        llvm::map_range(permAttr.getValues<APInt>(),
                        [](const APInt &val) { return val.getSExtValue(); }));

    SmallVector<int64_t> nonZeroPerms;
    nonZeroPerms.reserve(permValues.size());
    for (auto idx : permValues) {
      auto sz = inputTy.getDimSize(idx);
      if (sz != 1)
        nonZeroPerms.push_back(idx);
    }

    for (int i = 1, s = nonZeroPerms.size(); i < s; ++i)
      if (nonZeroPerms[i - 1] > nonZeroPerms[i])
        return rewriter.notifyMatchFailure(op,
                                           "Transpose changes memeory layout.");

    SmallVector<int64_t> newShape;
    newShape.reserve(inputTy.getRank());
    for (int i = 0, s = inputTy.getRank(); i < s; ++i)
      newShape.push_back(inputTy.getDimSize(permValues[i]));

    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
        op, op.getType(), op.getInput1(),
        rewriter.getDenseI64ArrayAttr(newShape));
    return success();
  }
};

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<ConsolidateTransposeOptimization, TransposeIsReshape>(context);
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

template <typename IntFolder, typename FloatFolder>
DenseElementsAttr binaryFolder(DenseElementsAttr lhs, DenseElementsAttr rhs,
                               RankedTensorType returnTy) {
  if (rhs && lhs && rhs.isSplat() && lhs.isSplat()) {
    auto lETy = lhs.getType().cast<ShapedType>().getElementType();
    auto rETy = rhs.getType().cast<ShapedType>().getElementType();
    if (lETy != rETy)
      return {};

    if (lETy.isa<IntegerType>()) {
      APInt l = lhs.getSplatValue<APInt>();
      APInt r = rhs.getSplatValue<APInt>();
      auto result = IntFolder()(l, r);
      return DenseElementsAttr::get(returnTy, result);
    }

    if (lETy.isa<FloatType>()) {
      APFloat l = lhs.getSplatValue<APFloat>();
      APFloat r = rhs.getSplatValue<APFloat>();
      auto result = FloatFolder()(l, r);
      return DenseElementsAttr::get(returnTy, result);
    }
  }

  return {};
}

OpFoldResult AddOp::fold(ArrayRef<Attribute> operands) {
  auto lhsTy = getInput1().getType().dyn_cast<RankedTensorType>();
  auto rhsTy = getInput2().getType().dyn_cast<RankedTensorType>();
  auto resultTy = getType().dyn_cast<RankedTensorType>();
  if (!lhsTy || !rhsTy || !resultTy)
    return {};
  if (lhsTy != rhsTy)
    return {};

  auto resultETy = resultTy.getElementType();
  auto lhsAttr = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsAttr = operands[1].dyn_cast_or_null<DenseElementsAttr>();

  if (lhsAttr && lhsAttr.isSplat() && resultETy.isa<FloatType>()) {
    if (lhsAttr.getSplatValue<APFloat>().isZero())
      return getInput2();
  }

  if (rhsAttr && rhsAttr.isSplat() && resultETy.isa<FloatType>()) {
    if (rhsAttr.getSplatValue<APFloat>().isZero())
      return getInput1();
  }

  if (lhsAttr && lhsAttr.isSplat() && resultETy.isa<IntegerType>()) {
    if (lhsAttr.getSplatValue<APInt>().isZero())
      return getInput2();
  }

  if (rhsAttr && rhsAttr.isSplat() && resultETy.isa<IntegerType>()) {
    if (rhsAttr.getSplatValue<APInt>().isZero())
      return getInput1();
  }

  if (!lhsAttr || !rhsAttr)
    return {};

  return binaryFolder<std::plus<APInt>, std::plus<APFloat>>(lhsAttr, rhsAttr,
                                                            lhsTy);
}

OpFoldResult DivOp::fold(ArrayRef<Attribute> operands) {
  auto lhsTy = getInput1().getType().dyn_cast<RankedTensorType>();
  auto rhsTy = getInput2().getType().dyn_cast<RankedTensorType>();
  auto resultTy = getType().dyn_cast<RankedTensorType>();
  if (!lhsTy || !rhsTy || !resultTy)
    return {};
  if (lhsTy != rhsTy)
    return {};

  auto resultETy = resultTy.getElementType();
  auto lhsAttr = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsAttr = operands[1].dyn_cast_or_null<DenseElementsAttr>();
  if (lhsAttr && lhsAttr.isSplat()) {
    if (resultETy.isa<IntegerType>() && lhsAttr.getSplatValue<APInt>().isZero())
      return lhsAttr;
  }

  if (rhsAttr && rhsAttr.isSplat()) {
    if (resultETy.isa<IntegerType>() && rhsAttr.getSplatValue<APInt>().isOne())
      return getInput1();
  }

  if (rhsAttr && lhsAttr && rhsAttr.isSplat() && lhsAttr.isSplat()) {
    if (resultETy.isa<IntegerType>()) {
      APInt l = lhsAttr.getSplatValue<APInt>();
      APInt r = rhsAttr.getSplatValue<APInt>();
      APInt result = l.sdiv(r);
      return DenseElementsAttr::get(resultTy, result);
    }
  }

  return {};
}

namespace {
DenseElementsAttr mulBinaryFolder(DenseElementsAttr lhs, DenseElementsAttr rhs,
                                  RankedTensorType ty, int32_t shift) {
  if (rhs && lhs && rhs.isSplat() && lhs.isSplat()) {
    if (ty.getElementType().isa<IntegerType>()) {
      APInt l = lhs.getSplatValue<APInt>();
      APInt r = rhs.getSplatValue<APInt>();

      if (shift == 0) {
        return DenseElementsAttr::get(ty, l * r);
      }

      auto bitwidth = ty.getElementType().getIntOrFloatBitWidth();
      l = l.sext(bitwidth * 2);
      r = r.sext(bitwidth * 2);
      auto result = l * r;
      result.lshrInPlace(shift);
      result = result.trunc(bitwidth);
      return DenseElementsAttr::get(ty, result);
    }

    if (ty.getElementType().isa<FloatType>()) {
      APFloat l = lhs.getSplatValue<APFloat>();
      APFloat r = rhs.getSplatValue<APFloat>();
      APFloat result = l * r;
      return DenseElementsAttr::get(ty, result);
    }
  }

  return {};
}
} // namespace

OpFoldResult MulOp::fold(ArrayRef<Attribute> operands) {
  auto lhs = getInput1();
  auto rhs = getInput2();
  auto lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhsTy = rhs.getType().dyn_cast<RankedTensorType>();
  auto resultTy = getType().dyn_cast<RankedTensorType>();
  if (!lhsTy || !rhsTy || !resultTy)
    return {};
  if (lhsTy != rhsTy)
    return {};

  auto resultETy = resultTy.getElementType();
  auto lhsAttr = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsAttr = operands[1].dyn_cast_or_null<DenseElementsAttr>();

  if (lhsAttr && lhsAttr.isSplat() && resultETy.isa<FloatType>()) {
    auto val = lhsAttr.getSplatValue<APFloat>();
    if (val.isZero())
      return lhsAttr;
    if (val.isExactlyValue(1.0))
      return rhs;
  }

  if (rhsAttr && rhsAttr.isSplat() && resultETy.isa<FloatType>()) {
    auto val = rhsAttr.getSplatValue<APFloat>();
    if (val.isZero())
      return rhsAttr;
    if (val.isExactlyValue(1.0))
      return lhs;
  }

  if (lhsAttr && lhsAttr.isSplat() && resultETy.isa<IntegerType>()) {
    auto val = lhsAttr.getSplatValue<APInt>();
    if (val.isZero())
      return lhsAttr;
    const int64_t shift = getShift();
    const int64_t shifted = 1LL << shift;
    if (val.getSExtValue() == shifted)
      return rhs;
  }

  if (rhsAttr && rhsAttr.isSplat() && resultETy.isa<IntegerType>()) {
    auto val = rhsAttr.getSplatValue<APInt>();
    const int64_t shift = getShift();
    const int64_t shifted = 1LL << shift;
    if (val.isZero())
      return rhsAttr;
    if (val.getSExtValue() == shifted)
      return lhs;
  }

  return mulBinaryFolder(lhsAttr, rhsAttr, lhsTy, getShift());
}

OpFoldResult SubOp::fold(ArrayRef<Attribute> operands) {
  auto lhsTy = getInput1().getType().dyn_cast<RankedTensorType>();
  auto rhsTy = getInput2().getType().dyn_cast<RankedTensorType>();
  auto resultTy = getType().dyn_cast<RankedTensorType>();
  if (!lhsTy || !rhsTy || !resultTy)
    return {};
  if (lhsTy != rhsTy)
    return {};

  auto resultETy = resultTy.getElementType();
  auto lhsAttr = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsAttr = operands[1].dyn_cast_or_null<DenseElementsAttr>();

  if (rhsAttr && rhsAttr.isSplat() && resultETy.isa<FloatType>()) {
    if (rhsAttr.getSplatValue<APFloat>().isZero())
      return getInput1();
  }

  if (rhsAttr && rhsAttr.isSplat() && resultETy.isa<IntegerType>()) {
    if (rhsAttr.getSplatValue<APInt>().isZero())
      return getInput1();
  }

  if (!lhsAttr || !rhsAttr)
    return {};

  return binaryFolder<std::minus<APInt>, std::minus<APFloat>>(lhsAttr, rhsAttr,
                                                              lhsTy);
}

namespace {
template <typename Cmp> struct ComparisonFold {
  ComparisonFold() = default;
  APInt operator()(const APInt &l, const APInt &r) {
    return APInt(1, Cmp()(l, r));
  }

  APInt operator()(const APFloat &l, const APFloat &r) {
    return APInt(1, Cmp()(l, r));
  }
};

struct APIntFoldGreater {
  APIntFoldGreater() = default;
  APInt operator()(const APInt &l, const APInt &r) {
    return APInt(1, l.sgt(r));
  }
};

struct APIntFoldGreaterEqual {
  APIntFoldGreaterEqual() = default;
  APInt operator()(const APInt &l, const APInt &r) {
    return APInt(1, l.sge(r));
  }
};
} // namespace

OpFoldResult GreaterOp::fold(ArrayRef<Attribute> operands) {
  auto resultTy = getType().dyn_cast<RankedTensorType>();
  auto lhsAttr = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsAttr = operands[1].dyn_cast_or_null<DenseElementsAttr>();

  if (!lhsAttr || !rhsAttr)
    return {};

  return binaryFolder<APIntFoldGreater, ComparisonFold<std::greater<APFloat>>>(
      lhsAttr, rhsAttr, resultTy);
}

OpFoldResult GreaterEqualOp::fold(ArrayRef<Attribute> operands) {
  auto resultTy = getType().dyn_cast<RankedTensorType>();
  auto lhsAttr = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsAttr = operands[1].dyn_cast_or_null<DenseElementsAttr>();

  if (!lhsAttr || !rhsAttr)
    return {};

  return binaryFolder<APIntFoldGreaterEqual,
                      ComparisonFold<std::greater_equal<APFloat>>>(
      lhsAttr, rhsAttr, resultTy);
}

OpFoldResult EqualOp::fold(ArrayRef<Attribute> operands) {
  auto resultTy = getType().dyn_cast<RankedTensorType>();
  auto lhsAttr = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsAttr = operands[1].dyn_cast_or_null<DenseElementsAttr>();
  Value lhs = getInput1();
  Value rhs = getInput2();
  auto lhsTy = lhs.getType().cast<ShapedType>();

  // If we are comparing an integer value to itself it is always true. We can
  // not do this with float due to float values.
  if (lhsTy.getElementType().isa<IntegerType>() && resultTy.hasStaticShape() &&
      lhs == rhs) {
    return DenseElementsAttr::get(resultTy, true);
  }

  if (!lhsAttr || !rhsAttr)
    return {};

  return binaryFolder<ComparisonFold<std::equal_to<APInt>>,
                      ComparisonFold<std::equal_to<APFloat>>>(lhsAttr, rhsAttr,
                                                              resultTy);
}

OpFoldResult CastOp::fold(ArrayRef<Attribute> operands) {
  if (getInput().getType() == getType())
    return getInput();

  auto operand = operands[0].dyn_cast_or_null<ElementsAttr>();
  if (!operand)
    return {};

  auto inTy = getInput().getType().cast<ShapedType>();
  auto outTy = getType().cast<ShapedType>();
  auto inETy = inTy.getElementType();
  auto outETy = outTy.getElementType();

  if (operand.isSplat()) {
    if (inETy.isa<FloatType>() && outETy.isa<FloatType>()) {
      bool overflow;
      auto splatVal = operand.getSplatValue<APFloat>();
      auto &semantics = outETy.cast<FloatType>().getFloatSemantics();
      splatVal.convert(semantics, llvm::RoundingMode::NearestTiesToEven,
                       &overflow);
      return SplatElementsAttr::get(outTy, splatVal);
    }

    if (inETy.isa<IntegerType>() && outETy.isa<FloatType>()) {
      auto unsign = inETy.cast<IntegerType>().isUnsignedInteger();
      APFloat splatVal(outETy.cast<FloatType>().getFloatSemantics());
      splatVal.convertFromAPInt(operand.getSplatValue<APInt>(), !unsign,
                                llvm::RoundingMode::NearestTiesToEven);
      return SplatElementsAttr::get(outTy, splatVal);
    }

    if (inETy.isa<FloatType>() && outETy.isa<IntegerType>()) {
      auto unsign = outETy.cast<IntegerType>().isUnsignedInteger();
      auto intVal =
          APSInt(outETy.cast<IntegerType>().getIntOrFloatBitWidth(), unsign);
      auto floatVal = operand.getSplatValue<APFloat>();
      bool exact;
      floatVal.convertToInteger(intVal, llvm::RoundingMode::TowardZero, &exact);
      return SplatElementsAttr::get(outTy, intVal);
    }

    if (inETy.isa<IntegerType>() && outETy.isa<IntegerType>()) {
      auto unsignIn = inETy.cast<IntegerType>().isUnsignedInteger();
      bool trunc =
          inETy.getIntOrFloatBitWidth() > outETy.getIntOrFloatBitWidth();
      auto intVal = operand.getSplatValue<APInt>();
      auto bitwidth = outETy.getIntOrFloatBitWidth();

      if (trunc) {
        intVal = intVal.trunc(bitwidth);
      } else if (unsignIn) {
        intVal = intVal.zext(bitwidth);
      } else {
        intVal = intVal.sext(bitwidth);
      }

      return SplatElementsAttr::get(outTy, intVal);
    }
  }

  return {};
}

OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValueAttr();
}

#define REDUCE_FOLDER(OP)                                                      \
  OpFoldResult OP::fold(ArrayRef<Attribute> operands) {                        \
    ShapedType inputTy = getInput().getType().cast<ShapedType>();              \
    if (!inputTy.hasRank())                                                    \
      return {};                                                               \
    if (inputTy.getDimSize(getAxis()) == 1)                                    \
      return getInput();                                                       \
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

  if (!inputTy || !outputTy)
    return {};

  if (inputTy == outputTy)
    return getInput1();

  auto operand = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  if (operand && outputTy.hasStaticShape() && operand.isSplat()) {
    return SplatElementsAttr::get(outputTy, operand.getSplatValue<Attribute>());
  }

  return {};
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

// Fold away cases where a tosa.resize operation returns a copy
// of the input image.
OpFoldResult ResizeOp::fold(ArrayRef<Attribute> operands) {
  ArrayRef<int64_t> offset = getOffset();
  ArrayRef<int64_t> border = getBorder();
  ArrayRef<int64_t> scale = getScale();

  // Check unit scaling.
  if (scale[0] != scale[1] || scale[2] != scale[3]) {
    return {};
  }

  // There should be no offset.
  if (offset[0] != 0 || offset[1] != 0) {
    return {};
  }

  // There should be no border.
  if (border[0] != 0 || border[1] != 0) {
    return {};
  }

  auto input = getInput();
  auto inputTy = input.getType().cast<RankedTensorType>();
  auto resultTy = getType().cast<RankedTensorType>();
  if (inputTy != resultTy)
    return {};

  return input;
}

OpFoldResult ReverseOp::fold(ArrayRef<Attribute> operands) {
  auto operand = getInput();
  auto operandTy = operand.getType().cast<ShapedType>();
  auto axis = getAxis();
  auto operandAttr = operands[0].dyn_cast_or_null<SplatElementsAttr>();
  if (operandAttr)
    return operandAttr;

  // If the dim-length is 1, tosa.reverse is a no-op.
  if (operandTy.hasRank() && operandTy.getDimSize(axis) == 1)
    return operand;

  return {};
}

OpFoldResult SliceOp::fold(ArrayRef<Attribute> operands) {
  auto inputTy = getInput().getType().dyn_cast<RankedTensorType>();
  auto outputTy = getType().dyn_cast<RankedTensorType>();

  if (!inputTy || !outputTy)
    return {};

  if (inputTy == outputTy && inputTy.hasStaticShape())
    return getInput();

  if (!operands[0])
    return {};

  auto operand = operands[0].cast<ElementsAttr>();
  if (operand.isSplat() && outputTy.hasStaticShape()) {
    return SplatElementsAttr::get(outputTy, operand.getSplatValue<Attribute>());
  }

  if (inputTy.hasStaticShape() && outputTy.hasStaticShape() &&
      outputTy.getNumElements() == 1) {
    llvm::SmallVector<uint64_t> indices(getStart());
    auto value = operand.getValues<Attribute>()[indices];
    return SplatElementsAttr::get(outputTy, value);
  }

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
  bool allOnes = llvm::all_of(getMultiples(), [](int64_t v) { return v == 1; });
  if (allOnes && getInput1().getType() == getType())
    return getInput1();
  return {};
}

OpFoldResult TransposeOp::fold(ArrayRef<Attribute> operands) {
  auto inputTy = getInput1().getType().cast<ShapedType>();
  auto resultTy = getType().cast<ShapedType>();

  // Transposing splat values just means reshaping.
  if (auto input = operands[0].dyn_cast_or_null<DenseElementsAttr>()) {
    if (input.isSplat() && resultTy.hasStaticShape() &&
        inputTy.getElementType() == resultTy.getElementType())
      return input.reshape(resultTy);
  }

  // Transpose does not change the input type.
  if (getInput1().getType() != getType())
    return {};

  // Transpose is not the identity transpose.
  SmallVector<int64_t> perms;
  if (getConstantPerms(perms).failed())
    return {};

  if (!llvm::equal(llvm::seq<int64_t>(0, perms.size()), perms))
    return {};

  return getInput1();
}
