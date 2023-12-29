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

    if (op.getInput1().getDefiningOp<tosa::TransposeOp>())
      return rewriter.notifyMatchFailure(
          op, "Src is from transpose, can compose transposes");

    Value result = op.getResult();
    for (Operation *subop : result.getUsers()) {
      if (dyn_cast_or_null<tosa::TransposeOp>(subop))
        return rewriter.notifyMatchFailure(
            op, "Dest is used by transpose, can compose transposes");
    }

    auto input = op.getInput1();
    auto inputTy = llvm::cast<ShapedType>(input.getType());
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
                                           "Transpose changes memory layout.");

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

struct MaterializePadValue : public OpRewritePattern<tosa::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::PadOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getPadConst())
      return failure();

    auto input = op.getInput1();
    auto padding = op.getPadding();

    ShapedType inputTy = llvm::cast<ShapedType>(input.getType());
    Type elementTy = inputTy.getElementType();

    Attribute constantAttr;
    if (llvm::isa<FloatType>(elementTy)) {
      constantAttr = rewriter.getFloatAttr(elementTy, 0.0);
    } else if (llvm::isa<IntegerType>(elementTy) && !op.getQuantizationInfo()) {
      constantAttr = rewriter.getIntegerAttr(elementTy, 0);
    } else if (llvm::isa<IntegerType>(elementTy) && op.getQuantizationInfo()) {
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
    ShapedType inputType = llvm::cast<ShapedType>(input.getType());
    ShapedType outputType = llvm::cast<ShapedType>(output.getType());

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
    auto inputType = llvm::dyn_cast<RankedTensorType>(op.getInput().getType());
    auto inputElementType = inputType.getElementType();

    if (!inputType.hasStaticShape()) {
      return failure();
    }

    if (inputElementType.isa<FloatType>()) {
      // Unlike integer types, floating point types can represent infinity.
      auto minClamp = op.getMinFp();
      auto maxClamp = op.getMaxFp();
      bool isMin = minClamp.isInfinity() && minClamp.isNegative();
      bool isMax = maxClamp.isInfinity() && !maxClamp.isNegative();

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

    if (llvm::isa<IntegerType>(inputElementType)) {
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

struct ConcatSliceOptimization : public OpRewritePattern<tosa::SliceOp> {
  using OpRewritePattern<tosa::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::SliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    Value sliceInput = sliceOp.getInput();
    auto concatOp = sliceInput.getDefiningOp<tosa::ConcatOp>();
    if (!concatOp)
      return rewriter.notifyMatchFailure(
          sliceOp, "slice input must be concat operation");

    OperandRange inputs = concatOp.getInput1();
    auto concatType = dyn_cast<RankedTensorType>(concatOp.getType());
    if (!concatType || !concatType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          sliceOp, "slice input must be a static ranked tensor");
    int32_t axis = concatOp.getAxis();

    llvm::SmallVector<int64_t> sliceStart(sliceOp.getStart());
    llvm::ArrayRef<int64_t> sliceSize = sliceOp.getSize();

    // Validate slice on the concatenated axis. Slicing along this
    // axis should span only one of the inputs to the concatenate
    // operation.
    std::optional<Value> replaceWithSlice;
    for (auto input : inputs) {
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      if (!inputType || !inputType.hasStaticShape())
        return rewriter.notifyMatchFailure(
            sliceOp, "concat input must be a static ranked tensor");

      if (sliceStart[axis] >= 0 &&
          (sliceStart[axis] + sliceSize[axis]) <= inputType.getDimSize(axis)) {
        replaceWithSlice = rewriter
                               .create<tosa::SliceOp>(
                                   sliceOp.getLoc(), sliceOp.getType(), input,
                                   rewriter.getDenseI64ArrayAttr(sliceStart),
                                   rewriter.getDenseI64ArrayAttr(sliceSize))
                               .getResult();
        break;
      }
      sliceStart[axis] -= inputType.getDimSize(axis);
    }

    if (!replaceWithSlice)
      return rewriter.notifyMatchFailure(
          sliceOp, "corresponding concat input not found for slice");

    rewriter.replaceOp(sliceOp, replaceWithSlice.value());
    return success();
  }
};

void SliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<ConcatSliceOptimization>(context);
}

//===----------------------------------------------------------------------===//
// Operator Folders.
//===----------------------------------------------------------------------===//

template <typename IntFolder, typename FloatFolder>
DenseElementsAttr binaryFolder(DenseElementsAttr lhs, DenseElementsAttr rhs,
                               RankedTensorType returnTy) {
  if (rhs && lhs && rhs.isSplat() && lhs.isSplat()) {
    auto lETy = llvm::cast<ShapedType>(lhs.getType()).getElementType();
    auto rETy = llvm::cast<ShapedType>(rhs.getType()).getElementType();
    if (lETy != rETy)
      return {};

    if (llvm::isa<IntegerType>(lETy)) {
      APInt l = lhs.getSplatValue<APInt>();
      APInt r = rhs.getSplatValue<APInt>();
      auto result = IntFolder()(l, r);
      return DenseElementsAttr::get(returnTy, result);
    }

    if (llvm::isa<FloatType>(lETy)) {
      APFloat l = lhs.getSplatValue<APFloat>();
      APFloat r = rhs.getSplatValue<APFloat>();
      auto result = FloatFolder()(l, r);
      return DenseElementsAttr::get(returnTy, result);
    }
  }

  return {};
}

static bool isSplatZero(Type elemType, DenseElementsAttr val) {
  if (llvm::isa<FloatType>(elemType))
    return val && val.isSplat() && val.getSplatValue<APFloat>().isZero();
  if (llvm::isa<IntegerType>(elemType))
    return val && val.isSplat() && val.getSplatValue<APInt>().isZero();
  return false;
}

static bool isSplatOne(Type elemType, DenseElementsAttr val, int64_t shift) {
  if (llvm::isa<FloatType>(elemType))
    return val && val.isSplat() &&
           val.getSplatValue<APFloat>().isExactlyValue(1.0);
  if (llvm::isa<IntegerType>(elemType)) {
    const int64_t shifted = 1LL << shift;
    return val && val.isSplat() &&
           val.getSplatValue<APInt>().getSExtValue() == shifted;
  }
  return false;
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto lhsTy = llvm::dyn_cast<RankedTensorType>(getInput1().getType());
  auto rhsTy = llvm::dyn_cast<RankedTensorType>(getInput2().getType());
  auto resultTy = llvm::dyn_cast<RankedTensorType>(getType());
  if (!lhsTy || !rhsTy || !resultTy)
    return {};

  auto resultETy = resultTy.getElementType();
  auto lhsAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1());
  auto rhsAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput2());

  if (lhsTy == resultTy && isSplatZero(resultETy, rhsAttr))
    return getInput1();
  if (rhsTy == resultTy && isSplatZero(resultETy, lhsAttr))
    return getInput2();

  if (!lhsAttr || !rhsAttr)
    return {};

  return binaryFolder<std::plus<APInt>, std::plus<APFloat>>(lhsAttr, rhsAttr,
                                                            resultTy);
}

OpFoldResult DivOp::fold(FoldAdaptor adaptor) {
  auto lhsTy = llvm::dyn_cast<RankedTensorType>(getInput1().getType());
  auto rhsTy = llvm::dyn_cast<RankedTensorType>(getInput2().getType());
  auto resultTy = llvm::dyn_cast<RankedTensorType>(getType());
  if (!lhsTy || !rhsTy || !resultTy)
    return {};
  if (lhsTy != rhsTy)
    return {};

  auto resultETy = resultTy.getElementType();
  auto lhsAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1());
  auto rhsAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput2());
  if (lhsAttr && lhsAttr.isSplat()) {
    if (llvm::isa<IntegerType>(resultETy) &&
        lhsAttr.getSplatValue<APInt>().isZero())
      return lhsAttr;
  }

  if (rhsAttr && rhsAttr.isSplat()) {
    if (llvm::isa<IntegerType>(resultETy) &&
        rhsAttr.getSplatValue<APInt>().isOne())
      return getInput1();
  }

  if (rhsAttr && lhsAttr && rhsAttr.isSplat() && lhsAttr.isSplat()) {
    if (llvm::isa<IntegerType>(resultETy)) {
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
    if (llvm::isa<IntegerType>(ty.getElementType())) {
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

    if (llvm::isa<FloatType>(ty.getElementType())) {
      APFloat l = lhs.getSplatValue<APFloat>();
      APFloat r = rhs.getSplatValue<APFloat>();
      APFloat result = l * r;
      return DenseElementsAttr::get(ty, result);
    }
  }

  return {};
}
} // namespace

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto lhs = getInput1();
  auto rhs = getInput2();
  auto lhsTy = llvm::dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsTy = llvm::dyn_cast<RankedTensorType>(rhs.getType());
  auto resultTy = llvm::dyn_cast<RankedTensorType>(getType());
  if (!lhsTy || !rhsTy || !resultTy)
    return {};

  auto resultETy = resultTy.getElementType();
  auto lhsAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1());
  auto rhsAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput2());

  const int64_t shift = llvm::isa<IntegerType>(resultETy) ? getShift() : 0;
  if (rhsTy == resultTy) {
    if (isSplatZero(resultETy, lhsAttr))
      return lhsAttr.resizeSplat(resultTy);
    if (isSplatOne(resultETy, lhsAttr, shift))
      return rhs;
  }
  if (lhsTy == resultTy) {
    if (isSplatZero(resultETy, rhsAttr))
      return rhsAttr.resizeSplat(resultTy);
    if (isSplatOne(resultETy, rhsAttr, shift))
      return lhs;
  }

  return mulBinaryFolder(lhsAttr, rhsAttr, resultTy, getShift());
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  auto lhsTy = llvm::dyn_cast<RankedTensorType>(getInput1().getType());
  auto rhsTy = llvm::dyn_cast<RankedTensorType>(getInput2().getType());
  auto resultTy = llvm::dyn_cast<RankedTensorType>(getType());
  if (!lhsTy || !rhsTy || !resultTy)
    return {};

  auto resultETy = resultTy.getElementType();
  auto lhsAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1());
  auto rhsAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput2());

  if (lhsTy == resultTy && isSplatZero(resultETy, rhsAttr))
    return getInput1();

  if (!lhsAttr || !rhsAttr)
    return {};

  return binaryFolder<std::minus<APInt>, std::minus<APFloat>>(lhsAttr, rhsAttr,
                                                              resultTy);
}

namespace {
template <typename Cmp>
struct ComparisonFold {
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

OpFoldResult GreaterOp::fold(FoldAdaptor adaptor) {
  auto resultTy = llvm::dyn_cast<RankedTensorType>(getType());
  auto lhsAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1());
  auto rhsAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput2());

  if (!lhsAttr || !rhsAttr)
    return {};

  return binaryFolder<APIntFoldGreater, ComparisonFold<std::greater<APFloat>>>(
      lhsAttr, rhsAttr, resultTy);
}

OpFoldResult GreaterEqualOp::fold(FoldAdaptor adaptor) {
  auto resultTy = llvm::dyn_cast<RankedTensorType>(getType());
  auto lhsAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1());
  auto rhsAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput2());

  if (!lhsAttr || !rhsAttr)
    return {};

  return binaryFolder<APIntFoldGreaterEqual,
                      ComparisonFold<std::greater_equal<APFloat>>>(
      lhsAttr, rhsAttr, resultTy);
}

OpFoldResult EqualOp::fold(FoldAdaptor adaptor) {
  auto resultTy = llvm::dyn_cast<RankedTensorType>(getType());
  auto lhsAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1());
  auto rhsAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput2());
  Value lhs = getInput1();
  Value rhs = getInput2();
  auto lhsTy = llvm::cast<ShapedType>(lhs.getType());

  // If we are comparing an integer value to itself it is always true. We can
  // not do this with float due to float values.
  if (llvm::isa<IntegerType>(lhsTy.getElementType()) && resultTy &&
      resultTy.hasStaticShape() && lhs == rhs) {
    return DenseElementsAttr::get(resultTy, true);
  }

  if (!lhsAttr || !rhsAttr)
    return {};

  return binaryFolder<ComparisonFold<std::equal_to<APInt>>,
                      ComparisonFold<std::equal_to<APFloat>>>(lhsAttr, rhsAttr,
                                                              resultTy);
}

OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  if (getInput().getType() == getType())
    return getInput();

  auto operand = llvm::dyn_cast_if_present<ElementsAttr>(adaptor.getInput());
  if (!operand)
    return {};

  auto inTy = llvm::cast<ShapedType>(getInput().getType());
  auto outTy = llvm::cast<ShapedType>(getType());
  auto inETy = inTy.getElementType();
  auto outETy = outTy.getElementType();

  if (operand.isSplat()) {
    if (llvm::isa<FloatType>(inETy) && llvm::isa<FloatType>(outETy)) {
      bool overflow;
      auto splatVal = operand.getSplatValue<APFloat>();
      auto &semantics = llvm::cast<FloatType>(outETy).getFloatSemantics();
      splatVal.convert(semantics, llvm::RoundingMode::NearestTiesToEven,
                       &overflow);
      return SplatElementsAttr::get(outTy, splatVal);
    }

    if (llvm::isa<IntegerType>(inETy) && llvm::isa<FloatType>(outETy)) {
      auto unsign = llvm::cast<IntegerType>(inETy).isUnsignedInteger();
      APFloat splatVal(llvm::cast<FloatType>(outETy).getFloatSemantics());
      splatVal.convertFromAPInt(operand.getSplatValue<APInt>(), !unsign,
                                llvm::RoundingMode::NearestTiesToEven);
      return SplatElementsAttr::get(outTy, splatVal);
    }

    if (llvm::isa<FloatType>(inETy) && llvm::isa<IntegerType>(outETy)) {
      auto unsign = llvm::cast<IntegerType>(outETy).isUnsignedInteger();
      auto intVal = APSInt(
          llvm::cast<IntegerType>(outETy).getIntOrFloatBitWidth(), unsign);
      auto floatVal = operand.getSplatValue<APFloat>();
      bool exact;
      floatVal.convertToInteger(intVal, llvm::RoundingMode::TowardZero, &exact);
      return SplatElementsAttr::get(outTy, intVal);
    }

    if (llvm::isa<IntegerType>(inETy) && llvm::isa<IntegerType>(outETy)) {
      auto unsignIn = llvm::cast<IntegerType>(inETy).isUnsignedInteger();
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

OpFoldResult ConstOp::fold(FoldAdaptor adaptor) { return getValueAttr(); }

#define REDUCE_FOLDER(OP)                                                      \
  OpFoldResult OP::fold(FoldAdaptor adaptor) {                                 \
    ShapedType inputTy = llvm::cast<ShapedType>(getInput().getType());         \
    if (!inputTy.hasRank())                                                    \
      return {};                                                               \
    if (inputTy != getType())                                                  \
      return {};                                                               \
    if (inputTy.getRank() == 0 || inputTy.getDimSize(getAxis()) == 1)          \
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

OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) {
  auto inputTy = llvm::dyn_cast<RankedTensorType>(getInput1().getType());
  auto outputTy = llvm::dyn_cast<RankedTensorType>(getType());

  if (!inputTy || !outputTy)
    return {};

  if (inputTy == outputTy)
    return getInput1();

  // reshape(reshape(x)) -> reshape(x)
  if (auto reshapeOp = llvm::dyn_cast_if_present<tosa::ReshapeOp>(
          getInput1().getDefiningOp())) {
    getInput1Mutable().assign(reshapeOp.getInput1());
    return getResult();
  }

  // reshape(const(x)) -> const(reshape-attr(x))
  if (auto operand = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1())) {
    // Constants must have static shape.
    if (!outputTy.hasStaticShape())
      return {};

    // Okay to duplicate splat constants.
    if (operand.isSplat())
      return SplatElementsAttr::get(outputTy, operand.getSplatValue<Attribute>());

    // Don't duplicate other constants.
    if (!getInput1().hasOneUse())
      return {};

    return operand.reshape(
        llvm::cast<ShapedType>(operand.getType()).clone(getNewShape()));
  }

  return {};
}

OpFoldResult PadOp::fold(FoldAdaptor adaptor) {
  // If the pad is all zeros we can fold this operation away.
  if (adaptor.getPadding()) {
    auto densePad = llvm::cast<DenseElementsAttr>(adaptor.getPadding());
    if (densePad.isSplat() && densePad.getSplatValue<APInt>().isZero()) {
      return getInput1();
    }
  }

  return {};
}

// Fold away cases where a tosa.resize operation returns a copy
// of the input image.
OpFoldResult ResizeOp::fold(FoldAdaptor adaptor) {
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
  auto inputTy = llvm::cast<RankedTensorType>(input.getType());
  auto resultTy = llvm::cast<RankedTensorType>(getType());
  if (inputTy != resultTy)
    return {};

  return input;
}

OpFoldResult ReverseOp::fold(FoldAdaptor adaptor) {
  auto operand = getInput();
  auto operandTy = llvm::cast<ShapedType>(operand.getType());
  auto axis = getAxis();
  auto operandAttr = llvm::dyn_cast_if_present<SplatElementsAttr>(adaptor.getInput());
  if (operandAttr)
    return operandAttr;

  // If the dim-length is 1, tosa.reverse is a no-op.
  if (operandTy.hasRank() &&
      (operandTy.getRank() == 0 || operandTy.getDimSize(axis) == 1))
    return operand;

  return {};
}

OpFoldResult SliceOp::fold(FoldAdaptor adaptor) {
  auto inputTy = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto outputTy = llvm::dyn_cast<RankedTensorType>(getType());

  if (!inputTy || !outputTy)
    return {};

  if (inputTy == outputTy && inputTy.hasStaticShape())
    return getInput();

  if (!adaptor.getInput())
    return {};

  // Cannot create an ElementsAttr from non-int/float/index types
  if (!inputTy.getElementType().isIntOrIndexOrFloat() ||
      !outputTy.getElementType().isIntOrIndexOrFloat())
    return {};

  auto operand = llvm::cast<ElementsAttr>(adaptor.getInput());
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

OpFoldResult tosa::SelectOp::fold(FoldAdaptor adaptor) {
  if (getOnTrue() == getOnFalse())
    return getOnTrue();

  auto predicate = llvm::dyn_cast_if_present<DenseIntElementsAttr>(adaptor.getPred());
  if (!predicate)
    return {};

  if (!predicate.isSplat())
    return {};
  return predicate.getSplatValue<APInt>().getBoolValue() ? getOnTrue()
                                                         : getOnFalse();
}

OpFoldResult TileOp::fold(FoldAdaptor adaptor) {
  bool allOnes = llvm::all_of(getMultiples(), [](int64_t v) { return v == 1; });
  if (allOnes && getInput1().getType() == getType())
    return getInput1();
  return {};
}

OpFoldResult TransposeOp::fold(FoldAdaptor adaptor) {
  auto inputTy = llvm::cast<ShapedType>(getInput1().getType());
  auto resultTy = llvm::cast<ShapedType>(getType());

  // Transposing splat values just means reshaping.
  if (auto input = llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getInput1())) {
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

OpFoldResult tosa::LogOp::fold(FoldAdaptor adaptor) {
  auto input = getInput1();
  // Element-wise log(exp(x)) = x
  if (auto op = input.getDefiningOp<tosa::ExpOp>()) {
    return op.getInput1();
  }

  return {};
}

OpFoldResult tosa::ExpOp::fold(FoldAdaptor adaptor) {
  auto input = getInput1();
  // Element-wise exp(log(x)) = x
  if (auto op = input.getDefiningOp<tosa::LogOp>()) {
    return op.getInput1();
  }

  return {};
}

OpFoldResult tosa::NegateOp::fold(FoldAdaptor adaptor) {
  auto input = getInput1();
  // Element-wise negate(negate(x)) = x
  if (auto op = input.getDefiningOp<tosa::NegateOp>()) {
    return op.getInput1();
  }

  return {};
}

OpFoldResult tosa::AbsOp::fold(FoldAdaptor adaptor) {
  auto input = getInput1();
  // Element-wise abs(abs(x)) = abs(x)
  if (auto op = input.getDefiningOp<tosa::AbsOp>()) {
    return input;
  }

  return {};
}

OpFoldResult ConcatOp::fold(FoldAdaptor adaptor) {
  // Fold consecutive concats on the same axis into a single op.
  // Keep track of the operands so we are able to construct a new concat
  // later. Conservatively assume that we double the number of operands when
  // folding
  SmallVector<Value, 8> concatOperands;
  concatOperands.reserve(2 * getNumOperands());

  // Find all operands that are foldable concats
  bool foundFoldableConcat = false;
  for (Value operand : getOperands()) {
    concatOperands.emplace_back(operand);

    auto producer = dyn_cast_or_null<ConcatOp>(operand.getDefiningOp());
    if (!producer)
      continue;

    // Not foldable if axes are not the same
    if (getAxis() != producer.getAxis())
      continue;

    // Replace the original operand with all incoming operands
    foundFoldableConcat = true;
    concatOperands.pop_back();
    llvm::append_range(concatOperands, producer->getOperands());
  }

  if (!foundFoldableConcat)
    return {};

  getOperation()->setOperands(concatOperands);
  return getResult();
}
