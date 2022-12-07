//===- TosaToLinalg.cpp - Lowering Tosa to Linalg Dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/CoversionUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

using namespace mlir;
using namespace mlir::tosa;

template <typename T>
static arith::ConstantOp
createConstFromIntAttribute(Operation *op, const std::string &attrName,
                            Type requiredAttrType, OpBuilder &rewriter) {
  auto castedN = static_cast<T>(
      op->getAttr(attrName).cast<IntegerAttr>().getValue().getSExtValue());
  return rewriter.create<arith::ConstantOp>(
      op->getLoc(), IntegerAttr::get(requiredAttrType, castedN));
}

static Value
createLinalgBodyCalculationForElementwiseOp(Operation *op, ValueRange args,
                                            ArrayRef<Type> resultTypes,
                                            PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  auto elementTy =
      op->getOperand(0).getType().cast<ShapedType>().getElementType();

  // tosa::AbsOp
  if (isa<tosa::AbsOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<math::AbsFOp>(loc, resultTypes, args);

  if (isa<tosa::AbsOp>(op) && elementTy.isa<IntegerType>()) {
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementTy));
    auto cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                              args[0], zero);
    auto neg = rewriter.create<arith::SubIOp>(loc, zero, args[0]);
    return rewriter.create<arith::SelectOp>(loc, cmp, args[0], neg);
  }

  // tosa::AddOp
  if (isa<tosa::AddOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<arith::AddFOp>(loc, resultTypes, args);

  if (isa<tosa::AddOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<arith::AddIOp>(loc, resultTypes, args);

  // tosa::SubOp
  if (isa<tosa::SubOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<arith::SubFOp>(loc, resultTypes, args);

  if (isa<tosa::SubOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<arith::SubIOp>(loc, resultTypes, args);

  // tosa::MulOp
  if (isa<tosa::MulOp>(op) && elementTy.isa<FloatType>()) {
    if (dyn_cast<tosa::MulOp>(op).getShift() != 0) {
      (void)rewriter.notifyMatchFailure(op,
                                        "Cannot have shift value for float");
      return nullptr;
    }
    return rewriter.create<arith::MulFOp>(loc, resultTypes, args);
  }

  // tosa::DivOp
  if (isa<tosa::DivOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<arith::DivSIOp>(loc, resultTypes, args);

  // tosa::ReciprocalOp
  if (isa<tosa::ReciprocalOp>(op) && elementTy.isa<FloatType>()) {
    auto one =
        rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(elementTy, 1));
    return rewriter.create<arith::DivFOp>(loc, resultTypes, one, args[0]);
  }

  if (isa<tosa::MulOp>(op) && elementTy.isa<IntegerType>()) {
    Value a = args[0];
    Value b = args[1];
    auto shift =
        op->getAttr("shift").cast<IntegerAttr>().getValue().getSExtValue();
    if (shift > 0) {
      auto shiftConst =
          rewriter.create<arith::ConstantIntOp>(loc, shift, /*bitwidth=*/8);
      if (!a.getType().isInteger(32))
        a = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI32Type(), a);

      if (!b.getType().isInteger(32))
        b = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI32Type(), b);

      auto result = rewriter.create<tosa::ApplyScaleOp>(
          loc, rewriter.getI32Type(), a, b, shiftConst,
          rewriter.getBoolAttr(false));

      if (elementTy.isInteger(32))
        return result;

      return rewriter.create<arith::TruncIOp>(loc, elementTy, result);
    }

    int aWidth = a.getType().getIntOrFloatBitWidth();
    int bWidth = b.getType().getIntOrFloatBitWidth();
    int cWidth = resultTypes[0].getIntOrFloatBitWidth();

    if (aWidth < cWidth)
      a = rewriter.create<arith::ExtSIOp>(loc, resultTypes[0], a);
    if (bWidth < cWidth)
      b = rewriter.create<arith::ExtSIOp>(loc, resultTypes[0], b);

    return rewriter.create<arith::MulIOp>(loc, resultTypes, a, b);
  }

  // tosa::NegateOp
  if (isa<tosa::NegateOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<arith::NegFOp>(loc, resultTypes, args);

  if (isa<tosa::NegateOp>(op) && elementTy.isa<IntegerType>() &&
      !cast<tosa::NegateOp>(op).getQuantizationInfo()) {
    auto constant =
        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(elementTy, 0));
    return rewriter.create<arith::SubIOp>(loc, resultTypes, constant, args[0]);
  }

  if (isa<tosa::NegateOp>(op) && elementTy.isa<IntegerType>() &&
      cast<tosa::NegateOp>(op).getQuantizationInfo()) {
    auto quantizationInfo = cast<tosa::NegateOp>(op).getQuantizationInfo();
    int32_t inputBitWidth = elementTy.getIntOrFloatBitWidth();
    int64_t inZp = quantizationInfo.value().getInputZp();
    int64_t outZp = quantizationInfo.value().getOutputZp();

    // Compute the maximum value that can occur in the intermediate buffer.
    int64_t zpAdd = inZp + outZp;
    int64_t maxValue = APInt::getSignedMaxValue(inputBitWidth).getSExtValue() +
                       std::abs(zpAdd) + 1;

    // Convert that maximum value into the maximum bitwidth needed to represent
    // it. We assume 48-bit numbers may be supported further in the pipeline.
    int intermediateBitWidth = 64;
    if (maxValue <= APInt::getSignedMaxValue(16).getSExtValue()) {
      intermediateBitWidth = 16;
    } else if (maxValue <= APInt::getSignedMaxValue(32).getSExtValue()) {
      intermediateBitWidth = 32;
    } else if (maxValue <= APInt::getSignedMaxValue(48).getSExtValue()) {
      intermediateBitWidth = 48;
    }

    Type intermediateType = rewriter.getIntegerType(intermediateBitWidth);
    Value zpAddValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(intermediateType, zpAdd));

    // The negation can be applied by doing:
    //  outputValue = inZp + outZp - inputValue
    auto ext = rewriter.create<arith::ExtSIOp>(loc, intermediateType, args[0]);
    auto sub = rewriter.create<arith::SubIOp>(loc, zpAddValue, ext);

    // Clamp to the negation range.
    auto min = rewriter.create<arith::ConstantIntOp>(
        loc, APInt::getSignedMinValue(inputBitWidth).getSExtValue(),
        intermediateType);
    auto max = rewriter.create<arith::ConstantIntOp>(
        loc, APInt::getSignedMaxValue(inputBitWidth).getSExtValue(),
        intermediateType);
    auto clamp = clampIntHelper(loc, sub, min, max, rewriter);

    // Truncate to the final value.
    return rewriter.create<arith::TruncIOp>(loc, elementTy, clamp);
  }

  // tosa::BitwiseAndOp
  if (isa<tosa::BitwiseAndOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<arith::AndIOp>(loc, resultTypes, args);

  // tosa::BitwiseOrOp
  if (isa<tosa::BitwiseOrOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<arith::OrIOp>(loc, resultTypes, args);

  // tosa::BitwiseNotOp
  if (isa<tosa::BitwiseNotOp>(op) && elementTy.isa<IntegerType>()) {
    auto allOnesAttr = rewriter.getIntegerAttr(
        elementTy, APInt::getAllOnes(elementTy.getIntOrFloatBitWidth()));
    auto allOnes = rewriter.create<arith::ConstantOp>(loc, allOnesAttr);
    return rewriter.create<arith::XOrIOp>(loc, resultTypes, args[0], allOnes);
  }

  // tosa::BitwiseXOrOp
  if (isa<tosa::BitwiseXorOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<arith::XOrIOp>(loc, resultTypes, args);

  // tosa::LogicalLeftShiftOp
  if (isa<tosa::LogicalLeftShiftOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<arith::ShLIOp>(loc, resultTypes, args);

  // tosa::LogicalRightShiftOp
  if (isa<tosa::LogicalRightShiftOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.create<arith::ShRUIOp>(loc, resultTypes, args);

  // tosa::ArithmeticRightShiftOp
  if (isa<tosa::ArithmeticRightShiftOp>(op) && elementTy.isa<IntegerType>()) {
    auto result = rewriter.create<arith::ShRSIOp>(loc, resultTypes, args);
    auto round = op->getAttr("round").cast<BoolAttr>().getValue();
    if (!round) {
      return result;
    }

    Type i1Ty = IntegerType::get(rewriter.getContext(), /*width=*/1);
    auto one =
        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(elementTy, 1));
    auto zero =
        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(elementTy, 0));
    auto i1one =
        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));

    // Checking that input2 != 0
    auto shiftValueGreaterThanZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, args[1], zero);

    // Checking for the last bit of input1 to be 1
    auto subtract =
        rewriter.create<arith::SubIOp>(loc, resultTypes, args[1], one);
    auto shifted =
        rewriter.create<arith::ShRSIOp>(loc, resultTypes, args[0], subtract)
            ->getResults();
    auto truncated =
        rewriter.create<arith::TruncIOp>(loc, i1Ty, shifted, std::nullopt);
    auto isInputOdd =
        rewriter.create<arith::AndIOp>(loc, i1Ty, truncated, i1one);

    auto shouldRound = rewriter.create<arith::AndIOp>(
        loc, i1Ty, shiftValueGreaterThanZero, isInputOdd);
    auto extended =
        rewriter.create<arith::ExtUIOp>(loc, resultTypes, shouldRound);
    return rewriter.create<arith::AddIOp>(loc, resultTypes, result, extended);
  }

  // tosa::ClzOp
  if (isa<tosa::ClzOp>(op) && elementTy.isa<IntegerType>()) {
    return rewriter.create<math::CountLeadingZerosOp>(loc, elementTy, args[0]);
  }

  // tosa::LogicalAnd
  if (isa<tosa::LogicalAndOp>(op) && elementTy.isInteger(1))
    return rewriter.create<arith::AndIOp>(loc, resultTypes, args);

  // tosa::LogicalNot
  if (isa<tosa::LogicalNotOp>(op) && elementTy.isInteger(1)) {
    auto one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(elementTy, 1));
    return rewriter.create<arith::XOrIOp>(loc, resultTypes, args[0], one);
  }

  // tosa::LogicalOr
  if (isa<tosa::LogicalOrOp>(op) && elementTy.isInteger(1))
    return rewriter.create<arith::OrIOp>(loc, resultTypes, args);

  // tosa::LogicalXor
  if (isa<tosa::LogicalXorOp>(op) && elementTy.isInteger(1))
    return rewriter.create<arith::XOrIOp>(loc, resultTypes, args);

  // tosa::PowOp
  if (isa<tosa::PowOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::math::PowFOp>(loc, resultTypes, args);

  // tosa::RsqrtOp
  if (isa<tosa::RsqrtOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::math::RsqrtOp>(loc, resultTypes, args);

  // tosa::LogOp
  if (isa<tosa::LogOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::math::LogOp>(loc, resultTypes, args);

  // tosa::ExpOp
  if (isa<tosa::ExpOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::math::ExpOp>(loc, resultTypes, args);

  // tosa::TanhOp
  if (isa<tosa::TanhOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<mlir::math::TanhOp>(loc, resultTypes, args);

  // tosa::GreaterOp
  if (isa<tosa::GreaterOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
                                          args[0], args[1]);

  if (isa<tosa::GreaterOp>(op) && elementTy.isSignlessInteger())
    return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                          args[0], args[1]);

  // tosa::GreaterEqualOp
  if (isa<tosa::GreaterEqualOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE,
                                          args[0], args[1]);

  if (isa<tosa::GreaterEqualOp>(op) && elementTy.isSignlessInteger())
    return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                          args[0], args[1]);

  // tosa::EqualOp
  if (isa<tosa::EqualOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                          args[0], args[1]);

  if (isa<tosa::EqualOp>(op) && elementTy.isSignlessInteger())
    return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                          args[0], args[1]);

  // tosa::SelectOp
  if (isa<tosa::SelectOp>(op)) {
    elementTy = op->getOperand(1).getType().cast<ShapedType>().getElementType();
    if (elementTy.isa<FloatType>() || elementTy.isa<IntegerType>())
      return rewriter.create<arith::SelectOp>(loc, args[0], args[1], args[2]);
  }

  // tosa::MaximumOp
  if (isa<tosa::MaximumOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.create<arith::MaxFOp>(loc, args[0], args[1]);
  }

  if (isa<tosa::MaximumOp>(op) && elementTy.isSignlessInteger()) {
    auto predicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, args[0], args[1]);
    return rewriter.create<arith::SelectOp>(loc, predicate, args[0], args[1]);
  }

  // tosa::MinimumOp
  if (isa<tosa::MinimumOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.create<arith::MinFOp>(loc, args[0], args[1]);
  }

  if (isa<tosa::MinimumOp>(op) && elementTy.isSignlessInteger()) {
    auto predicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, args[0], args[1]);
    return rewriter.create<arith::SelectOp>(loc, predicate, args[0], args[1]);
  }

  // tosa::CeilOp
  if (isa<tosa::CeilOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<math::CeilOp>(loc, resultTypes, args);

  // tosa::FloorOp
  if (isa<tosa::FloorOp>(op) && elementTy.isa<FloatType>())
    return rewriter.create<math::FloorOp>(loc, resultTypes, args);

  // tosa::ClampOp
  if (isa<tosa::ClampOp>(op) && elementTy.isa<FloatType>()) {
    bool losesInfo = false;
    APFloat minApf = op->getAttr("min_fp").cast<FloatAttr>().getValue();
    APFloat maxApf = op->getAttr("max_fp").cast<FloatAttr>().getValue();
    minApf.convert(elementTy.cast<FloatType>().getFloatSemantics(),
                   APFloat::rmNearestTiesToEven, &losesInfo);
    maxApf.convert(elementTy.cast<FloatType>().getFloatSemantics(),
                   APFloat::rmNearestTiesToEven, &losesInfo);
    auto min = rewriter.create<arith::ConstantOp>(
        loc, elementTy, rewriter.getFloatAttr(elementTy, minApf));
    auto max = rewriter.create<arith::ConstantOp>(
        loc, elementTy, rewriter.getFloatAttr(elementTy, maxApf));
    return clampFloatHelper(loc, args[0], min, max, rewriter);
  }

  if (isa<tosa::ClampOp>(op) && elementTy.isa<IntegerType>()) {
    auto intTy = elementTy.cast<IntegerType>();
    int32_t min = static_cast<int32_t>(
        op->getAttr("min_int").cast<IntegerAttr>().getValue().getSExtValue());
    int32_t max = static_cast<int32_t>(
        op->getAttr("max_int").cast<IntegerAttr>().getValue().getSExtValue());

    if (intTy.isUnsignedInteger()) {
      min = std::max<int32_t>(min, 0);
      max = std::min<int32_t>(
          max,
          APInt::getMaxValue(intTy.getIntOrFloatBitWidth()).getSExtValue());
    } else {
      min = std::max<int32_t>(
          min, APInt::getSignedMinValue(intTy.getIntOrFloatBitWidth())
                   .getSExtValue());
      max = std::min<int32_t>(
          max, APInt::getSignedMaxValue(intTy.getIntOrFloatBitWidth())
                   .getSExtValue());
    }

    auto minVal = rewriter.create<arith::ConstantIntOp>(
        loc, min, intTy.getIntOrFloatBitWidth());
    auto maxVal = rewriter.create<arith::ConstantIntOp>(
        loc, max, intTy.getIntOrFloatBitWidth());
    return clampIntHelper(loc, args[0], minVal, maxVal, rewriter);
  }

  // tosa::SigmoidOp
  if (isa<tosa::SigmoidOp>(op) && elementTy.isa<FloatType>()) {
    auto one =
        rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(elementTy, 1));
    auto negate = rewriter.create<arith::NegFOp>(loc, resultTypes, args[0]);
    auto exp = rewriter.create<mlir::math::ExpOp>(loc, resultTypes, negate);
    auto added = rewriter.create<arith::AddFOp>(loc, resultTypes, exp, one);
    return rewriter.create<arith::DivFOp>(loc, resultTypes, one, added);
  }

  // tosa::CastOp
  if (isa<tosa::CastOp>(op)) {
    Type srcTy = elementTy;
    Type dstTy = resultTypes.front();
    bool bitExtend =
        srcTy.getIntOrFloatBitWidth() < dstTy.getIntOrFloatBitWidth();

    if (srcTy == dstTy)
      return args.front();

    if (srcTy.isa<FloatType>() && dstTy.isa<FloatType>() && bitExtend)
      return rewriter.create<arith::ExtFOp>(loc, resultTypes, args,
                                            std::nullopt);

    if (srcTy.isa<FloatType>() && dstTy.isa<FloatType>() && !bitExtend)
      return rewriter.create<arith::TruncFOp>(loc, resultTypes, args,
                                              std::nullopt);

    // 1-bit integers need to be treated as signless.
    if (srcTy.isInteger(1) && arith::UIToFPOp::areCastCompatible(srcTy, dstTy))
      return rewriter.create<arith::UIToFPOp>(loc, resultTypes, args,
                                              std::nullopt);

    if (srcTy.isInteger(1) && dstTy.isa<IntegerType>() && bitExtend)
      return rewriter.create<arith::ExtUIOp>(loc, resultTypes, args,
                                             std::nullopt);

    // Unsigned integers need an unrealized cast so that they can be passed
    // to UIToFP.
    if (srcTy.isUnsignedInteger() && dstTy.isa<FloatType>()) {
      auto unrealizedCast =
          rewriter
              .create<UnrealizedConversionCastOp>(
                  loc, rewriter.getIntegerType(srcTy.getIntOrFloatBitWidth()),
                  args[0])
              .getResult(0);
      return rewriter.create<arith::UIToFPOp>(loc, resultTypes[0],
                                              unrealizedCast);
    }

    // All other si-to-fp conversions should be handled by SIToFP.
    if (arith::SIToFPOp::areCastCompatible(srcTy, dstTy))
      return rewriter.create<arith::SIToFPOp>(loc, resultTypes, args,
                                              std::nullopt);

    // Casting to boolean, floats need to only be checked as not-equal to zero.
    if (srcTy.isa<FloatType>() && dstTy.isInteger(1)) {
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(srcTy, 0.0));
      return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE,
                                            args.front(), zero);
    }

    if (arith::FPToSIOp::areCastCompatible(srcTy, dstTy)) {
      auto zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(0.0f));
      auto half = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(0.5f));

      auto intMin = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(
                   APInt::getSignedMinValue(dstTy.getIntOrFloatBitWidth())
                       .getSExtValue()));

      auto intMax = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(
                   APInt::getSignedMaxValue(dstTy.getIntOrFloatBitWidth())
                       .getSExtValue()));

      auto added = rewriter.create<arith::AddFOp>(loc, args[0], half);
      auto subbed = rewriter.create<arith::SubFOp>(loc, args[0], half);
      auto negative = rewriter.create<arith::CmpFOp>(
          loc, arith::CmpFPredicate::OLT, args[0], zero);
      auto rounded =
          rewriter.create<arith::SelectOp>(loc, negative, subbed, added);

      auto clamped = clampFloatHelper(loc, rounded, intMin, intMax, rewriter);

      return rewriter.create<arith::FPToSIOp>(loc, dstTy, clamped);
    }

    // Casting to boolean, integers need to only be checked as not-equal to
    // zero.
    if (srcTy.isa<IntegerType>() && dstTy.isInteger(1)) {
      Value zero = rewriter.create<arith::ConstantIntOp>(
          loc, 0, srcTy.getIntOrFloatBitWidth());
      return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                            args.front(), zero);
    }

    if (srcTy.isa<IntegerType>() && dstTy.isa<IntegerType>() && bitExtend)
      return rewriter.create<arith::ExtSIOp>(loc, resultTypes, args,
                                             std::nullopt);

    if (srcTy.isa<IntegerType>() && dstTy.isa<IntegerType>() && !bitExtend) {
      auto intMin = rewriter.create<arith::ConstantIntOp>(
          loc,
          APInt::getSignedMinValue(dstTy.getIntOrFloatBitWidth())
              .getSExtValue(),
          srcTy.getIntOrFloatBitWidth());

      auto intMax = rewriter.create<arith::ConstantIntOp>(
          loc,
          APInt::getSignedMaxValue(dstTy.getIntOrFloatBitWidth())
              .getSExtValue(),
          srcTy.getIntOrFloatBitWidth());

      auto clamped = clampIntHelper(loc, args[0], intMin, intMax, rewriter);
      return rewriter.create<arith::TruncIOp>(loc, dstTy, clamped);
    }
  }

  (void)rewriter.notifyMatchFailure(
      op, "unhandled op for linalg body calculation for elementwise op");
  return nullptr;
}

static LogicalResult
elementwiseMatchAndRewriteHelper(Operation *operation,
                                 PatternRewriter &rewriter) {
  auto loc = operation->getLoc();

  assert(operation->getNumResults() == 1 &&
         "All TOSA elementwise ops should only return a single result.");

  auto results = operation->getResults();
  auto resultTy = operation->getResult(0).getType().dyn_cast<ShapedType>();

  if (!resultTy)
    return rewriter.notifyMatchFailure(operation,
                                       "All results must be a shaped type");

  unsigned rank = resultTy.getRank();

  // Construct the indexing maps needed for linalg.generic ops.
  SmallVector<Type> bodyArgTypes;

  for (Value in : operation->getOperands())
    bodyArgTypes.emplace_back(getElementTypeOrSelf(in.getType()));

  SmallVector<Type> opResultTypes;
  SmallVector<Value> emptyTensors;

  SmallVector<Value> dynDims;
  dynDims.resize(results.front().getType().cast<ShapedType>().getRank());

  for (auto arg : operation->getOperands()) {
    auto operandTy = arg.getType().cast<ShapedType>();
    for (int i = 0; i < operandTy.getRank(); i++) {
      if (operandTy.isDynamicDim(i) && !dynDims[i])
        dynDims[i] = rewriter.create<tensor::DimOp>(loc, arg, i);
    }
  }

  SmallVector<Value> filteredDims = condenseValues(dynDims);

  for (auto result : results) {
    auto resultTy = result.getType().template cast<ShapedType>();
    emptyTensors.push_back(rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultTy.getElementType(), filteredDims));
    opResultTypes.push_back(result.getType());
  }

  auto bodyResultTypes = llvm::to_vector<4>(llvm::map_range(
      emptyTensors, [](Value v) { return getElementTypeOrSelf(v); }));

  SmallVector<Value, 2> operands;
  SmallVector<AffineMap, 2> indexingMaps;
  indexingMaps.reserve(operation->getNumOperands() + bodyResultTypes.size());

  // Input indexing maps may be broadcasted.
  for (Value operand : operation->getOperands()) {
    ShapedType type = operand.getType().cast<ShapedType>();

    if (type.getShape() == resultTy.getShape()) {
      operands.push_back(operand);
      indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
      continue;
    }

    SmallVector<int64_t, 5> newShape;
    SmallVector<AffineExpr, 4> affineExprs;
    newShape.reserve(type.getRank());
    for (const auto &it : llvm::enumerate(type.getShape())) {
      if (it.value() == resultTy.getDimSize(it.index())) {
        newShape.push_back(it.value());
        affineExprs.push_back(
            mlir::getAffineDimExpr(it.index(), rewriter.getContext()));
      }
    }

    if (newShape.size() != rank) {
      operand = rewriter.create<tosa::ReshapeOp>(
          loc, RankedTensorType::get(newShape, type.getElementType()), operand,
          rewriter.getI64ArrayAttr(newShape));
    }

    operands.push_back(operand);
    indexingMaps.push_back(AffineMap::get(
        /*dimCount=*/rank, /*symbolCount=*/0, affineExprs,
        rewriter.getContext()));
  }

  indexingMaps.append(operation->getNumResults(),
                      rewriter.getMultiDimIdentityMap(rank));

  bool didEncounterError = false;
  auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc, opResultTypes, operands, emptyTensors, indexingMaps,
      getNParallelLoopsAttrs(rank),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value opResult = createLinalgBodyCalculationForElementwiseOp(
            operation, blockArgs.take_front(operation->getNumOperands()),
            bodyResultTypes, rewriter);
        if (!opResult) {
          didEncounterError = true;
          return;
        }
        nestedBuilder.create<linalg::YieldOp>(loc, opResult);
      });

  if (didEncounterError)
    return failure();

  rewriter.replaceOp(operation, linalgOp->getResults());
  return success();
}

// Returns the constant initial value for a given reduction operation. The
// attribute type varies depending on the element type required.
static Attribute createInitialValueForReduceOp(Operation *op, Type elementTy,
                                               PatternRewriter &rewriter) {
  if (isa<tosa::ReduceSumOp>(op) && elementTy.isa<FloatType>())
    return rewriter.getFloatAttr(elementTy, 0.0);

  if (isa<tosa::ReduceSumOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.getIntegerAttr(elementTy, 0);

  if (isa<tosa::ReduceProdOp>(op) && elementTy.isa<FloatType>())
    return rewriter.getFloatAttr(elementTy, 1.0);

  if (isa<tosa::ReduceProdOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.getIntegerAttr(elementTy, 1);

  if (isa<tosa::ReduceMinOp>(op) && elementTy.isa<FloatType>())
    return rewriter.getFloatAttr(
        elementTy, APFloat::getLargest(
                       elementTy.cast<FloatType>().getFloatSemantics(), false));

  if (isa<tosa::ReduceMinOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.getIntegerAttr(
        elementTy, APInt::getSignedMaxValue(elementTy.getIntOrFloatBitWidth()));

  if (isa<tosa::ReduceMaxOp>(op) && elementTy.isa<FloatType>())
    return rewriter.getFloatAttr(
        elementTy, APFloat::getLargest(
                       elementTy.cast<FloatType>().getFloatSemantics(), true));

  if (isa<tosa::ReduceMaxOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.getIntegerAttr(
        elementTy, APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth()));

  if (isa<tosa::ReduceAllOp>(op) && elementTy.isInteger(1))
    return rewriter.getIntegerAttr(elementTy, APInt::getAllOnes(1));

  if (isa<tosa::ReduceAnyOp>(op) && elementTy.isInteger(1))
    return rewriter.getIntegerAttr(elementTy, APInt::getZero(1));

  if (isa<tosa::ArgMaxOp>(op) && elementTy.isa<FloatType>())
    return rewriter.getFloatAttr(
        elementTy, APFloat::getLargest(
                       elementTy.cast<FloatType>().getFloatSemantics(), true));

  if (isa<tosa::ArgMaxOp>(op) && elementTy.isa<IntegerType>())
    return rewriter.getIntegerAttr(
        elementTy, APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth()));

  return {};
}

// Creates the body calculation for a reduction. The operations vary depending
// on the input type.
static Value createLinalgBodyCalculationForReduceOp(Operation *op,
                                                    ValueRange args,
                                                    Type elementTy,
                                                    PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  if (isa<tosa::ReduceSumOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.create<arith::AddFOp>(loc, args);
  }

  if (isa<tosa::ReduceSumOp>(op) && elementTy.isa<IntegerType>()) {
    return rewriter.create<arith::AddIOp>(loc, args);
  }

  if (isa<tosa::ReduceProdOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.create<arith::MulFOp>(loc, args);
  }

  if (isa<tosa::ReduceProdOp>(op) && elementTy.isa<IntegerType>()) {
    return rewriter.create<arith::MulIOp>(loc, args);
  }

  if (isa<tosa::ReduceMinOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.create<arith::MinFOp>(loc, args[0], args[1]);
  }

  if (isa<tosa::ReduceMinOp>(op) && elementTy.isa<IntegerType>()) {
    auto predicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, args[0], args[1]);
    return rewriter.create<arith::SelectOp>(loc, predicate, args[0], args[1]);
  }

  if (isa<tosa::ReduceMaxOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.create<arith::MaxFOp>(loc, args[0], args[1]);
  }

  if (isa<tosa::ReduceMaxOp>(op) && elementTy.isa<IntegerType>()) {
    auto predicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, args[0], args[1]);
    return rewriter.create<arith::SelectOp>(loc, predicate, args[0], args[1]);
  }

  if (isa<tosa::ReduceAllOp>(op) && elementTy.isInteger(1))
    return rewriter.create<arith::AndIOp>(loc, args);

  if (isa<tosa::ReduceAnyOp>(op) && elementTy.isInteger(1))
    return rewriter.create<arith::OrIOp>(loc, args);

  return {};
}

// Performs the match and rewrite for reduction operations. This includes
// declaring a correctly sized initial value, and the linalg.generic operation
// that reduces across the specified axis.
static LogicalResult reduceMatchAndRewriteHelper(Operation *op, uint64_t axis,
                                                 PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto inputTy = op->getOperand(0).getType().template cast<ShapedType>();
  auto resultTy = op->getResult(0).getType().template cast<ShapedType>();
  auto elementTy = resultTy.getElementType();
  Value input = op->getOperand(0);

  llvm::SmallVector<int64_t> reduceShape;
  SmallVector<Value> dynDims;
  for (unsigned i = 0; i < inputTy.getRank(); i++) {
    if (axis != i) {
      reduceShape.push_back(inputTy.getDimSize(i));
      if (inputTy.isDynamicDim(i))
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
    }
  }

  Type reduceTy = RankedTensorType::get(reduceShape, resultTy.getElementType());

  // First fill the output buffer with the init value.
  auto emptyTensor =
      rewriter
          .create<tensor::EmptyOp>(loc, reduceShape, resultTy.getElementType(),
                                   dynDims)
          .getResult();

  auto fillValueAttr = createInitialValueForReduceOp(op, elementTy, rewriter);
  if (!fillValueAttr)
    return rewriter.notifyMatchFailure(
        op, "No initial value found for reduction operation");

  auto fillValue = rewriter.create<arith::ConstantOp>(loc, fillValueAttr);
  auto filledTensor = rewriter
                          .create<linalg::FillOp>(loc, ValueRange{fillValue},
                                                  ValueRange{emptyTensor})
                          .result();

  SmallVector<AffineExpr, 2> srcExprs;
  SmallVector<AffineExpr, 2> dstExprs;
  SmallVector<utils::IteratorType, 4> iteratorTypes;
  for (unsigned int i = 0, rank = inputTy.getRank(); i != rank; ++i) {
    srcExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));

    iteratorTypes.push_back(axis == i ? utils::IteratorType::reduction
                                      : utils::IteratorType::parallel);
    if (axis != i)
      dstExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
  }

  bool didEncounterError = false;
  auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs});
  auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc, reduceTy, input, filledTensor, maps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto result = createLinalgBodyCalculationForReduceOp(
            op, blockArgs, elementTy, rewriter);
        if (result)
          didEncounterError = true;

        nestedBuilder.create<linalg::YieldOp>(loc, result);
      });

  if (!didEncounterError)
    return failure();

  SmallVector<ReassociationExprs, 4> reassociationMap;
  uint64_t expandInputRank =
      linalgOp.getResults()[0].getType().cast<ShapedType>().getRank();
  reassociationMap.resize(expandInputRank);

  for (uint64_t i = 0; i < expandInputRank; i++) {
    int32_t dimToPush = i > axis ? i + 1 : i;
    reassociationMap[i].push_back(rewriter.getAffineDimExpr(dimToPush));
  }

  if (expandInputRank != 0) {
    int32_t expandedDim = axis < expandInputRank ? axis : expandInputRank - 1;
    reassociationMap[expandedDim].push_back(
        rewriter.getAffineDimExpr(expandedDim + 1));
  }

  rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
      op, resultTy, linalgOp.getResults()[0], reassociationMap);
  return success();
}

static bool findIntermediateShape(ArrayRef<int64_t> lhsShape,
                                  ArrayRef<int64_t> rhsShape,
                                  SmallVector<int64_t> &intermediateShape,
                                  bool isDynamic) {
  if (isDynamic) {
    // TODO (natashaknk): Make dynamic intermediate shape not always be rank-1
    intermediateShape = {ShapedType::kDynamic};
    return true;
  }

  if (lhsShape.empty() || rhsShape.empty()) {
    intermediateShape = {};
    return true;
  }

  unsigned currLhsDim = 0, currRhsDim = 0;
  while (currLhsDim < lhsShape.size() && currRhsDim < rhsShape.size()) {
    int64_t rhsSize = rhsShape[currRhsDim];
    int64_t lhsSize = lhsShape[currLhsDim];
    while (lhsSize != rhsSize && currLhsDim < lhsShape.size() &&
           currRhsDim < rhsShape.size()) {
      if (lhsSize < rhsSize) {
        currLhsDim++;
        lhsSize *= lhsShape[currLhsDim];
      } else {
        currRhsDim++;
        rhsSize *= rhsShape[currRhsDim];
      }
    }
    if (lhsSize == rhsSize) {
      intermediateShape.push_back(lhsSize);
    }
    currRhsDim++;
    currLhsDim++;
  }

  // If the iterators didn't reach the end and their leftover dimensions are not
  // equal to 1 an intermediate shape was not found.
  while (currLhsDim < lhsShape.size()) {
    if (lhsShape[currLhsDim++] != 1) {
      return false;
    }
  }

  while (currRhsDim < rhsShape.size()) {
    if (rhsShape[currRhsDim++] != 1) {
      return false;
    }
  }

  return true;
}

static bool createReassociationMapsForCollapse(
    PatternRewriter &rewriter, ArrayRef<int64_t> srcShape,
    ArrayRef<int64_t> dstShape,
    SmallVector<ReassociationExprs, 4> &reassociationMap, bool isDynamic) {

  // If the shape is dynamic, create a map for collapsing into one dimension.
  if (isDynamic) {
    SmallVector<AffineExpr, 2> exprs;
    for (int i = 0, s = srcShape.size(); i < s; ++i)
      exprs.push_back(rewriter.getAffineDimExpr(i));
    reassociationMap = {exprs};
    return true;
  }

  if (dstShape.empty()) {
    reassociationMap = {};
    return true;
  }

  reassociationMap.resize(dstShape.size());
  unsigned currSrcDim = 0, currDstDim = 0;
  while (currSrcDim < srcShape.size() && currDstDim < dstShape.size()) {
    int64_t dstSize = dstShape[currDstDim];
    int64_t srcSize = srcShape[currSrcDim];
    while (srcSize < dstSize && currSrcDim < srcShape.size()) {
      reassociationMap[currDstDim].push_back(
          rewriter.getAffineDimExpr(currSrcDim++));
      srcSize *= srcShape[currSrcDim];
    }
    if (srcSize == dstSize) {
      reassociationMap[currDstDim].push_back(
          rewriter.getAffineDimExpr(currSrcDim++));
      // If the next dim in collapsedShape is not 1, treat subsequent dims in
      // expandedShape which are 1 to be collapsed.
      if (currDstDim == dstShape.size() - 1 || dstShape[currDstDim + 1] != 1) {
        while (currSrcDim < srcShape.size() && srcShape[currSrcDim] == 1) {
          reassociationMap[currDstDim].push_back(
              rewriter.getAffineDimExpr(currSrcDim++));
        }
      }
    }
    currDstDim++;
  }

  // If both iterators didn't reach the end, we have leftover dimentions which
  // implies that we have a mismatch in shape.
  return currSrcDim == srcShape.size() && currDstDim == dstShape.size();
}

namespace {

template <typename SrcOp>
class PointwiseConverter : public OpRewritePattern<SrcOp> {
public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const final {
    return elementwiseMatchAndRewriteHelper(op, rewriter);
  }
};

class ReshapeConverterCollapse : public OpConversionPattern<tosa::ReshapeOp> {
public:
  using OpConversionPattern<tosa::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ReshapeOp reshape, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    ShapedType operandTy = adaptor.getInput1().getType().cast<ShapedType>();
    ShapedType resultTy = reshape.getType().template cast<ShapedType>();
    bool isDynamic = !operandTy.hasStaticShape();

    if (isDynamic && resultTy.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          reshape, "Cannot collapse dynamic dims to more than one dimension");
    }

    if (operandTy == resultTy) {
      rewriter.replaceOp(reshape, adaptor.getOperands()[0]);
      return success();
    }

    SmallVector<ReassociationExprs, 4> reassociationMap;
    if (!createReassociationMapsForCollapse(rewriter, operandTy.getShape(),
                                            resultTy.getShape(),
                                            reassociationMap, isDynamic)) {
      return rewriter.notifyMatchFailure(
          reshape,
          "tosa.reshape Attempting to collapse into an incompatible shape");
    }

    SmallVector<int64_t> intermediateShape;
    if (!findIntermediateShape(operandTy.getShape(), resultTy.getShape(),
                               intermediateShape, isDynamic)) {
      return rewriter.notifyMatchFailure(
          reshape, "tosa.reshape Cannot collapse into given shape");
    }

    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        reshape, resultTy, adaptor.getOperands()[0], reassociationMap);
    return success();
  }
};

class ReshapeConverterExpand : public OpConversionPattern<tosa::ReshapeOp> {
public:
  using OpConversionPattern<tosa::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ReshapeOp reshape, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    ShapedType operandTy = adaptor.getInput1().getType().cast<ShapedType>();
    ShapedType resultTy = reshape.getType().template cast<ShapedType>();
    bool isDynamic = !operandTy.hasStaticShape();

    if (operandTy == resultTy) {
      rewriter.replaceOp(reshape, adaptor.getOperands()[0]);
      return success();
    }

    if (isDynamic && operandTy.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          reshape, "Cannot expand dynamic dims from more than one dimension");
    }

    SmallVector<ReassociationExprs, 4> reassociationMap;
    if (!createReassociationMapsForCollapse(rewriter, resultTy.getShape(),
                                            operandTy.getShape(),
                                            reassociationMap, isDynamic)) {
      return rewriter.notifyMatchFailure(
          reshape,
          "tosa.reshape Attempting to expand into an incompatible shape");
    }

    SmallVector<int64_t> intermediateShape;
    if (!findIntermediateShape(operandTy.getShape(), resultTy.getShape(),
                               intermediateShape, isDynamic) ||
        intermediateShape != operandTy.getShape()) {
      return rewriter.notifyMatchFailure(
          reshape, "tosa.reshape Cannot expand into given shape");
    }
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        reshape, resultTy, adaptor.getOperands()[0], reassociationMap);
    return success();
  }
};

class ReshapeConverterCollapseExpand
    : public OpConversionPattern<tosa::ReshapeOp> {
public:
  using OpConversionPattern<tosa::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ReshapeOp reshape, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    ShapedType operandTy = adaptor.getInput1().getType().cast<ShapedType>();
    ShapedType resultTy = reshape.getType().template cast<ShapedType>();
    bool isDynamic = !operandTy.hasStaticShape();

    if (operandTy == resultTy) {
      rewriter.replaceOp(reshape, adaptor.getOperands()[0]);
      return success();
    }

    SmallVector<int64_t> intermediateShape;
    if (!findIntermediateShape(resultTy.getShape(), operandTy.getShape(),
                               intermediateShape, isDynamic)) {
      return rewriter.notifyMatchFailure(
          reshape, "tosa.reshape Cannot identify an intermediate shape between "
                   "the given two shapes");
    }

    Value collapse = rewriter.create<tosa::ReshapeOp>(
        reshape.getLoc(),
        RankedTensorType::get(intermediateShape,
                              reshape.getType().getElementType()),
        adaptor.getInput1());
    Value expand =
        rewriter.create<tosa::ReshapeOp>(reshape.getLoc(), resultTy, collapse);
    rewriter.replaceOp(reshape, expand);

    return success();
  }
};

class TransposeConverter : public OpRewritePattern<tosa::TransposeOp> {
public:
  using OpRewritePattern<tosa::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::TransposeOp op,
                                PatternRewriter &rewriter) const final {
    DenseIntElementsAttr perms;
    if (!matchPattern(op.getPerms(), m_Constant(&perms))) {
      return failure();
    }

    auto loc = op.getLoc();
    auto input = op->getOperand(0);
    auto resultTy = op.getType().cast<ShapedType>();

    SmallVector<Value> dynDims;
    dynDims.resize(op->getResult(0).getType().cast<ShapedType>().getRank());

    SmallVector<AffineExpr, 2> inputExprs;
    inputExprs.resize(resultTy.getRank());
    auto operandTy = input.getType().cast<ShapedType>();
    for (const auto &permutation : llvm::enumerate(perms.getValues<APInt>())) {
      auto index = permutation.index();
      auto value = permutation.value().getZExtValue();
      if (!operandTy.hasRank() || operandTy.isDynamicDim(index)) {
        dynDims[value] = rewriter.create<tensor::DimOp>(loc, input, index);
      }
      inputExprs[value] = rewriter.getAffineDimExpr(index);
    }

    SmallVector<Value> filteredDims = condenseValues(dynDims);

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultTy.getElementType(), filteredDims);

    SmallVector<AffineMap, 2> affineMaps = {
        AffineMap::get(resultTy.getRank(), /*symbolCount=*/0, inputExprs,
                       rewriter.getContext()),
        rewriter.getMultiDimIdentityMap(resultTy.getRank())};

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, resultTy, op.getInput1(), ValueRange{emptyTensor}, affineMaps,
        getNParallelLoopsAttrs(resultTy.getRank()),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(loc, *args.begin());
        });
    return success();
  }
};

class RescaleConverter : public OpRewritePattern<tosa::RescaleOp> {
public:
  using OpRewritePattern<tosa::RescaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::RescaleOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto inputTy = op.getInput().getType().cast<ShapedType>();
    auto outputTy = op.getOutput().getType().cast<ShapedType>();
    unsigned rank = inputTy.getRank();

    // This is an illegal configuration. terminate and log an error
    if (op.getDoubleRound() && !op.getScale32())
      return rewriter.notifyMatchFailure(
          op, "tosa.rescale requires scale32 for double_round to be true");

    SmallVector<Value> dynDims;
    for (int i = 0; i < outputTy.getRank(); i++) {
      if (outputTy.isDynamicDim(i)) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
      }
    }

    // The shift and multiplier values.
    SmallVector<int32_t> multiplierValues;
    getValuesFromIntArrayAttribute(op.getMultiplier(), multiplierValues);

    SmallVector<int8_t> shiftValues;
    getValuesFromIntArrayAttribute(op.getShift(), shiftValues);

    // If we shift by more than the bitwidth, this just sets to 0.
    for (int i = 0, s = multiplierValues.size(); i < s; i++) {
      if (shiftValues[i] > 63) {
        shiftValues[i] = 0;
        multiplierValues[i] = 0;
      }
    }

    // Double round only occurs if shift is greater than 31, check that this
    // is ever true.
    bool doubleRound =
        op.getDoubleRound() &&
        llvm::any_of(shiftValues, [](int32_t v) { return v > 31; });

    SmallVector<AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(rank)};
    SmallVector<Value, 4> genericInputs = {input};

    // If we are rescaling per-channel then we need to store the multiplier
    // values in a buffer.
    Value multiplierConstant;
    int64_t multiplierArg = 0;
    if (multiplierValues.size() == 1) {
      multiplierConstant = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(multiplierValues.front()));
    } else {
      SmallVector<AffineExpr, 2> multiplierExprs{
          rewriter.getAffineDimExpr(rank - 1)};
      auto multiplierType =
          RankedTensorType::get({static_cast<int64_t>(multiplierValues.size())},
                                rewriter.getI32Type());
      genericInputs.push_back(rewriter.create<arith::ConstantOp>(
          loc, DenseIntElementsAttr::get(multiplierType, multiplierValues)));

      indexingMaps.push_back(AffineMap::get(/*dimCount=*/rank,
                                            /*symbolCount=*/0, multiplierExprs,
                                            rewriter.getContext()));

      multiplierArg = indexingMaps.size() - 1;
    }

    // If we are rescaling per-channel then we need to store the shift
    // values in a buffer.
    Value shiftConstant;
    int64_t shiftArg = 0;
    if (shiftValues.size() == 1) {
      shiftConstant = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI8IntegerAttr(shiftValues.front()));
    } else {
      SmallVector<AffineExpr, 2> shiftExprs = {
          rewriter.getAffineDimExpr(rank - 1)};
      auto shiftType =
          RankedTensorType::get({static_cast<int64_t>(shiftValues.size())},
                                rewriter.getIntegerType(8));
      genericInputs.push_back(rewriter.create<arith::ConstantOp>(
          loc, DenseIntElementsAttr::get(shiftType, shiftValues)));
      indexingMaps.push_back(AffineMap::get(/*dimCount=*/rank,
                                            /*symbolCount=*/0, shiftExprs,
                                            rewriter.getContext()));
      shiftArg = indexingMaps.size() - 1;
    }

    // Indexing maps for output values.
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));

    // Construct the indexing maps needed for linalg.generic ops.
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, outputTy.getShape(), outputTy.getElementType(),
        ArrayRef<Value>({dynDims}));

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, outputTy, genericInputs, ValueRange{emptyTensor}, indexingMaps,
        getNParallelLoopsAttrs(rank),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value value = blockArgs[0];
          Type valueTy = value.getType();

          // For now we do all of our math in 64-bit. This is not optimal but
          // should be correct for now, consider computing correct bit depth
          // later.
          int32_t inBitwidth = valueTy.getIntOrFloatBitWidth() > 32 ? 48 : 32;

          auto inputZp = createConstFromIntAttribute<int32_t>(
              op, "input_zp", nestedBuilder.getIntegerType(inBitwidth),
              nestedBuilder);
          auto outputZp = createConstFromIntAttribute<int32_t>(
              op, "output_zp", nestedBuilder.getI32Type(), nestedBuilder);

          Value multiplier = multiplierConstant ? multiplierConstant
                                                : blockArgs[multiplierArg];
          Value shift = shiftConstant ? shiftConstant : blockArgs[shiftArg];

          if (valueTy.getIntOrFloatBitWidth() < 32) {
            if (valueTy.isUnsignedInteger()) {
              value = nestedBuilder
                          .create<UnrealizedConversionCastOp>(
                              nestedLoc,
                              nestedBuilder.getIntegerType(
                                  valueTy.getIntOrFloatBitWidth()),
                              value)
                          .getResult(0);
              value = nestedBuilder.create<arith::ExtUIOp>(
                  nestedLoc, nestedBuilder.getI32Type(), value);
            } else {
              value = nestedBuilder.create<arith::ExtSIOp>(
                  nestedLoc, nestedBuilder.getI32Type(), value);
            }
          }

          value =
              nestedBuilder.create<arith::SubIOp>(nestedLoc, value, inputZp);

          value = nestedBuilder.create<tosa::ApplyScaleOp>(
              loc, nestedBuilder.getI32Type(), value, multiplier, shift,
              nestedBuilder.getBoolAttr(doubleRound));

          // Move to the new zero-point.
          value =
              nestedBuilder.create<arith::AddIOp>(nestedLoc, value, outputZp);

          // Saturate to the output size.
          IntegerType outIntType =
              blockArgs.back().getType().cast<IntegerType>();
          unsigned outBitWidth = outIntType.getWidth();

          int32_t intMin = APInt::getSignedMinValue(outBitWidth).getSExtValue();
          int32_t intMax = APInt::getSignedMaxValue(outBitWidth).getSExtValue();

          // Unsigned integers have a difference output value.
          if (outIntType.isUnsignedInteger()) {
            intMin = 0;
            intMax = APInt::getMaxValue(outBitWidth).getZExtValue();
          }

          auto intMinVal = nestedBuilder.create<arith::ConstantOp>(
              loc, nestedBuilder.getI32IntegerAttr(intMin));
          auto intMaxVal = nestedBuilder.create<arith::ConstantOp>(
              loc, nestedBuilder.getI32IntegerAttr(intMax));

          value = clampIntHelper(nestedLoc, value, intMinVal, intMaxVal,
                                 nestedBuilder);

          if (outIntType.getWidth() < 32) {
            value = nestedBuilder.create<arith::TruncIOp>(
                nestedLoc, rewriter.getIntegerType(outIntType.getWidth()),
                value);

            if (outIntType.isUnsignedInteger()) {
              value = nestedBuilder
                          .create<UnrealizedConversionCastOp>(nestedLoc,
                                                              outIntType, value)
                          .getResult(0);
            }
          }

          nestedBuilder.create<linalg::YieldOp>(loc, value);
        });

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

// Handle the case where the resize operation is a regular broadcast. We
// perform this part separately to avoid generating Extract operations which
// are difficult to vectorize / optimize.
class BroadcastResizeConverter : public OpRewritePattern<tosa::ResizeOp> {
public:
  using OpRewritePattern<tosa::ResizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ResizeOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder builder(loc, rewriter);
    auto input = op.getInput();
    auto inputTy = input.getType().cast<RankedTensorType>();
    auto resultTy = op.getType().cast<RankedTensorType>();

    auto imageH = inputTy.getDimSize(1);
    auto imageW = inputTy.getDimSize(2);

    if (imageH != 1 || imageW != 1) {
      return rewriter.notifyMatchFailure(
          op, "tosa.resize is not a pure broadcast operation");
    }

    // TODO(suderman): These string values should be declared the TOSA dialect.
    if (op.getMode() != "NEAREST_NEIGHBOR" && op.getMode() != "BILINEAR")
      return failure();

    const bool isBilinear = op.getMode() == "BILINEAR";

    SmallVector<int32_t> scale;
    getValuesFromIntArrayAttribute(op.getScale(), scale);

    // Collapse the 1 dimensions away.
    SmallVector<ReassociationExprs, 4> collapseMap(2);
    collapseMap[0].push_back(builder.getAffineDimExpr(0));
    collapseMap[1].push_back(builder.getAffineDimExpr(1));
    collapseMap[1].push_back(builder.getAffineDimExpr(2));
    collapseMap[1].push_back(builder.getAffineDimExpr(3));

    auto collapseTy =
        RankedTensorType::get({inputTy.getDimSize(0), inputTy.getDimSize(3)},
                              inputTy.getElementType());
    Value collapse =
        builder.create<tensor::CollapseShapeOp>(collapseTy, input, collapseMap);

    // Broadcast input to the output shape.
    llvm::SmallVector<Value> outputDynSize;
    if (inputTy.isDynamicDim(0))
      outputDynSize.push_back(builder.create<tensor::DimOp>(input, 0));

    if (inputTy.isDynamicDim(3))
      outputDynSize.push_back(builder.create<tensor::DimOp>(input, 3));

    llvm::SmallVector<AffineExpr> inputExprs{
        rewriter.getAffineDimExpr(0),
        rewriter.getAffineDimExpr(3),
    };

    auto inputMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0,
                                   inputExprs, builder.getContext());
    auto resultMap = rewriter.getMultiDimIdentityMap(resultTy.getRank());
    SmallVector<utils::IteratorType> iterators(4,
                                               utils::IteratorType::parallel);

    Value empty = builder.create<tensor::EmptyOp>(
        resultTy.getShape(), resultTy.getElementType(), outputDynSize);

    auto generic = builder.create<linalg::GenericOp>(
        resultTy, ValueRange{collapse}, ValueRange{empty},
        ArrayRef<AffineMap>{inputMap, resultMap}, iterators,
        [=](OpBuilder &b, Location loc, ValueRange args) {
          Value value = args[0];
          // This is the quantized case.
          if (inputTy.getElementType() != resultTy.getElementType()) {
            value =
                b.create<arith::ExtSIOp>(loc, resultTy.getElementType(), value);

            if (isBilinear && scale[0] != 0) {
              Value scaleY = b.create<arith::ConstantOp>(
                  loc, b.getI32IntegerAttr(scale[0]));
              value = b.create<arith::MulIOp>(loc, value, scaleY);
            }

            if (isBilinear && scale[2] != 0) {
              Value scaleX = b.create<arith::ConstantOp>(
                  loc, b.getI32IntegerAttr(scale[2]));
              value = b.create<arith::MulIOp>(loc, value, scaleX);
            }
          }

          b.create<linalg::YieldOp>(loc, value);
        });

    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

class GenericResizeConverter : public OpRewritePattern<tosa::ResizeOp> {
public:
  using OpRewritePattern<tosa::ResizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ResizeOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto input = op.getInput();
    auto inputTy = input.getType().cast<ShapedType>();
    auto resultTy = op.getType().cast<ShapedType>();
    auto resultElementTy = resultTy.getElementType();

    auto imageH = inputTy.getShape()[1];
    auto imageW = inputTy.getShape()[2];

    auto dynamicDimsOr =
        checkHasDynamicBatchDims(rewriter, op, {input, op.getOutput()});
    if (!dynamicDimsOr.has_value())
      return failure();
    SmallVector<Value> dynamicDims = dynamicDimsOr.value();

    if (op.getMode() != "NEAREST_NEIGHBOR" && op.getMode() != "BILINEAR")
      return failure();

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultTy.getShape(), resultElementTy, dynamicDims);

    SmallVector<AffineMap, 2> affineMaps = {
        rewriter.getMultiDimIdentityMap(resultTy.getRank())};

    Value resize = input;
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultTy, ValueRange({}), ValueRange{emptyTensor}, affineMaps,
        getNParallelLoopsAttrs(resultTy.getRank()));
    resize = genericOp.getResult(0);

    OpBuilder::InsertionGuard regionGuard(rewriter);
    rewriter.createBlock(&genericOp.getRegion(), genericOp.getRegion().end(),
                         TypeRange({resultElementTy}), loc);
    Value batch = rewriter.create<linalg::IndexOp>(loc, 0);
    Value y = rewriter.create<linalg::IndexOp>(loc, 1);
    Value x = rewriter.create<linalg::IndexOp>(loc, 2);
    Value channel = rewriter.create<linalg::IndexOp>(loc, 3);

    auto hwMin =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
    auto hMax = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(imageH - 1));
    auto wMax = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(imageW - 1));

    Value inY =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), y);
    Value inX =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), x);

    bool floatingPointMode = resultElementTy.isF32();

    Value yScaleN, yScaleD, xScaleN, xScaleD, yOffset, xOffset, yBorder,
        xBorder;
    SmallVector<int32_t> scale, offset, border;
    getValuesFromIntArrayAttribute(op.getScale(), scale);
    getValuesFromIntArrayAttribute(op.getOffset(), offset);
    getValuesFromIntArrayAttribute(op.getBorder(), border);

    yScaleN = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(scale[0]));
    yScaleD = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(scale[1]));
    xScaleN = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(scale[2]));
    xScaleD = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(scale[3]));
    yOffset = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(offset[0]));
    xOffset = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(offset[1]));
    yBorder = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(border[0]));
    xBorder = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(border[1]));

    // Compute the the integer index and partial offset.
    Value ix, iy, dx, dy;
    // x = x * scale_d + offset;
    // ix = floor(x / scale_n)
    if (floatingPointMode) {
      // dx = x / scale_n - ix
      Value y =
          rewriter.create<arith::UIToFPOp>(loc, rewriter.getF32Type(), inY);
      Value x =
          rewriter.create<arith::UIToFPOp>(loc, rewriter.getF32Type(), inX);

      yScaleN =
          rewriter.create<arith::UIToFPOp>(loc, rewriter.getF32Type(), yScaleN);
      yScaleD =
          rewriter.create<arith::UIToFPOp>(loc, rewriter.getF32Type(), yScaleD);
      xScaleN =
          rewriter.create<arith::UIToFPOp>(loc, rewriter.getF32Type(), xScaleN);
      xScaleD =
          rewriter.create<arith::UIToFPOp>(loc, rewriter.getF32Type(), xScaleD);
      yOffset =
          rewriter.create<arith::UIToFPOp>(loc, rewriter.getF32Type(), yOffset);
      xOffset =
          rewriter.create<arith::UIToFPOp>(loc, rewriter.getF32Type(), xOffset);

      y = rewriter.create<arith::MulFOp>(loc, y, yScaleD);
      x = rewriter.create<arith::MulFOp>(loc, x, xScaleD);

      y = rewriter.create<arith::AddFOp>(loc, y, yOffset);
      x = rewriter.create<arith::AddFOp>(loc, x, xOffset);

      y = rewriter.create<arith::DivFOp>(loc, y, yScaleN);
      x = rewriter.create<arith::DivFOp>(loc, x, xScaleN);

      iy = rewriter.create<math::FloorOp>(loc, y);
      ix = rewriter.create<math::FloorOp>(loc, x);

      dy = rewriter.create<arith::SubFOp>(loc, y, iy);
      dx = rewriter.create<arith::SubFOp>(loc, x, ix);

      iy = rewriter.create<arith::FPToSIOp>(loc, rewriter.getI32Type(), iy);
      ix = rewriter.create<arith::FPToSIOp>(loc, rewriter.getI32Type(), ix);
    } else {
      //  dx = x - ix * scale_n;
      Value y = rewriter.create<arith::MulIOp>(loc, inY, yScaleD);
      Value x = rewriter.create<arith::MulIOp>(loc, inX, xScaleD);

      y = rewriter.create<arith::AddIOp>(loc, y, yOffset);
      x = rewriter.create<arith::AddIOp>(loc, x, xOffset);

      iy = rewriter.create<arith::DivUIOp>(loc, y, yScaleN);
      ix = rewriter.create<arith::DivUIOp>(loc, x, xScaleN);

      Value tempY = rewriter.create<arith::MulIOp>(loc, iy, yScaleN);
      Value tempX = rewriter.create<arith::MulIOp>(loc, ix, xScaleN);

      dy = rewriter.create<arith::SubIOp>(loc, y, tempY);
      dx = rewriter.create<arith::SubIOp>(loc, x, tempX);
    }

    if (op.getMode() == "NEAREST_NEIGHBOR") {
      Value yPred, xPred;
      auto zeroVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(0));
      auto oneVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(1));

      // Round the index position towards the closest pixel location.
      if (floatingPointMode) {
        auto halfVal = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getF32FloatAttr(0.5f));
        yPred = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE,
                                               dy, halfVal);
        xPred = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE,
                                               dx, halfVal);
      } else {
        Value yScaleNHalfVal =
            rewriter.create<arith::ShRSIOp>(loc, yScaleN, oneVal);
        Value xScaleNHalfVal =
            rewriter.create<arith::ShRSIOp>(loc, xScaleN, oneVal);
        yPred = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                               dy, yScaleNHalfVal);
        xPred = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                               dx, xScaleNHalfVal);
      }

      auto yOffset =
          rewriter.create<arith::SelectOp>(loc, yPred, oneVal, zeroVal);
      auto xOffset =
          rewriter.create<arith::SelectOp>(loc, xPred, oneVal, zeroVal);

      iy = rewriter.create<arith::AddIOp>(loc, iy, yOffset);
      ix = rewriter.create<arith::AddIOp>(loc, ix, xOffset);

      // Clamp the to be within the bounds of the input image.
      iy = clampIntHelper(loc, iy, hwMin, hMax, rewriter);
      ix = clampIntHelper(loc, ix, hwMin, wMax, rewriter);

      // Read the value from the input array.
      iy =
          rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), iy);
      ix =
          rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), ix);

      Value result = rewriter.create<tensor::ExtractOp>(
          loc, input, ValueRange{batch, iy, ix, channel});

      rewriter.create<linalg::YieldOp>(loc, result);
    } else {
      // The mode here must be BILINEAR.
      assert(op.getMode() == "BILINEAR");
      Value y0 = iy;
      Value x0 = ix;

      auto oneVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(1));
      Value y1 = rewriter.create<arith::AddIOp>(loc, y0, oneVal);
      Value x1 = rewriter.create<arith::AddIOp>(loc, x0, oneVal);

      y0 = clampIntHelper(loc, y0, hwMin, hMax, rewriter);
      y1 = clampIntHelper(loc, y1, hwMin, hMax, rewriter);

      x0 = clampIntHelper(loc, x0, hwMin, wMax, rewriter);
      x1 = clampIntHelper(loc, x1, hwMin, wMax, rewriter);

      y0 =
          rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), y0);
      y1 =
          rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), y1);
      x0 =
          rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), x0);
      x1 =
          rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), x1);

      Value y0x0 = rewriter.create<tensor::ExtractOp>(
          loc, input, ValueRange{batch, y0, x0, channel});
      Value y0x1 = rewriter.create<tensor::ExtractOp>(
          loc, input, ValueRange{batch, y0, x1, channel});
      Value y1x0 = rewriter.create<tensor::ExtractOp>(
          loc, input, ValueRange{batch, y1, x0, channel});
      Value y1x1 = rewriter.create<tensor::ExtractOp>(
          loc, input, ValueRange{batch, y1, x1, channel});

      if (floatingPointMode) {
        Value rightPart = dx;
        auto oneVal = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getF32FloatAttr(1.0f));
        Value leftPart = rewriter.create<arith::SubFOp>(loc, oneVal, dx);

        y0x0 = rewriter.create<arith::MulFOp>(loc, y0x0, leftPart);
        y0x1 = rewriter.create<arith::MulFOp>(loc, y0x1, rightPart);
        Value topAcc = rewriter.create<arith::AddFOp>(loc, y0x0, y0x1);

        y1x0 = rewriter.create<arith::MulFOp>(loc, y1x0, leftPart);
        y1x1 = rewriter.create<arith::MulFOp>(loc, y1x1, rightPart);
        Value bottomAcc = rewriter.create<arith::AddFOp>(loc, y1x0, y1x1);

        Value bottomPart = dy;
        Value topPart = rewriter.create<arith::SubFOp>(loc, oneVal, dy);
        topAcc = rewriter.create<arith::MulFOp>(loc, topAcc, topPart);
        bottomAcc = rewriter.create<arith::MulFOp>(loc, bottomAcc, bottomPart);
        Value result = rewriter.create<arith::AddFOp>(loc, topAcc, bottomAcc);

        rewriter.create<linalg::YieldOp>(loc, result);
      } else {
        // Perform in quantized space.
        y0x0 = rewriter.create<arith::ExtSIOp>(loc, resultElementTy, y0x0);
        y0x1 = rewriter.create<arith::ExtSIOp>(loc, resultElementTy, y0x1);
        y1x0 = rewriter.create<arith::ExtSIOp>(loc, resultElementTy, y1x0);
        y1x1 = rewriter.create<arith::ExtSIOp>(loc, resultElementTy, y1x1);

        if (resultElementTy.getIntOrFloatBitWidth() > 32) {
          dx = rewriter.create<arith::ExtSIOp>(loc, resultElementTy, dx);
          dy = rewriter.create<arith::ExtSIOp>(loc, resultElementTy, dy);
        }

        Value topAcc, bottomAcc;
        if (imageW == 1) {
          topAcc = rewriter.create<arith::MulIOp>(loc, y0x0, xScaleN);
          bottomAcc = rewriter.create<arith::MulIOp>(loc, y1x0, xScaleN);
        } else {
          Value rightPart = dx;
          Value leftPart = rewriter.create<arith::SubIOp>(loc, xScaleN, dx);

          y0x0 = rewriter.create<arith::MulIOp>(loc, y0x0, leftPart);
          y0x1 = rewriter.create<arith::MulIOp>(loc, y0x1, rightPart);
          topAcc = rewriter.create<arith::AddIOp>(loc, y0x0, y0x1);

          y1x0 = rewriter.create<arith::MulIOp>(loc, y1x0, leftPart);
          y1x1 = rewriter.create<arith::MulIOp>(loc, y1x1, rightPart);
          bottomAcc = rewriter.create<arith::AddIOp>(loc, y1x0, y1x1);
        }

        Value result;
        if (imageH == 1) {
          result = rewriter.create<arith::MulIOp>(loc, topAcc, yScaleN);
        } else {
          Value bottomPart = dy;
          Value topPart = rewriter.create<arith::SubIOp>(loc, yScaleN, dy);
          topAcc = rewriter.create<arith::MulIOp>(loc, topAcc, topPart);
          bottomAcc =
              rewriter.create<arith::MulIOp>(loc, bottomAcc, bottomPart);
          result = rewriter.create<arith::AddIOp>(loc, topAcc, bottomAcc);
        }

        rewriter.create<linalg::YieldOp>(loc, result);
      }
    }

    rewriter.replaceOp(op, resize);
    return success();
  }
};

// At the codegen level any identity operations should be removed. Any cases
// where identity is load-bearing (e.g. cross device computation) should be
// handled before lowering to codegen.
template <typename SrcOp>
class IdentityNConverter : public OpRewritePattern<SrcOp> {
public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, op.getOperation()->getOperands());
    return success();
  }
};

template <typename SrcOp>
class ReduceConverter : public OpRewritePattern<SrcOp> {
public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp reduceOp,
                                PatternRewriter &rewriter) const final {
    return reduceMatchAndRewriteHelper(reduceOp, reduceOp.getAxis(), rewriter);
  }
};

struct ConcatConverter : public OpConversionPattern<tosa::ConcatOp> {
  using OpConversionPattern<tosa::ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = op.getOperand(0).getType().template cast<ShapedType>();
    auto resultType = op.getType().dyn_cast<RankedTensorType>();

    Location loc = op.getLoc();
    int axis = op.getAxis();
    Value axisValue = rewriter.createOrFold<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(axis));
    int rank = resultType.getRank();
    SmallVector<Value, 3> offsets, sizes, strides;
    sizes.reserve(rank);
    strides.resize(rank, rewriter.create<arith::ConstantIndexOp>(loc, 1));
    offsets.resize(rank, rewriter.create<arith::ConstantIndexOp>(loc, 0));

    SmallVector<Value> dynDims;
    for (int i = 0; i < rank; ++i) {
      sizes.push_back(rewriter.createOrFold<tensor::DimOp>(
          loc, adaptor.getOperands()[0], i));
      if (inputType.isDynamicDim(i)) {
        dynDims.push_back(
            rewriter.create<tensor::DimOp>(loc, op.getOperand(0), i));
      }
    }

    Value resultDimSize = sizes[axis];
    for (auto arg : adaptor.getOperands().drop_front()) {
      auto size = rewriter.createOrFold<tensor::DimOp>(loc, arg, axisValue);
      resultDimSize =
          rewriter.createOrFold<arith::AddIOp>(loc, resultDimSize, size);
    }
    sizes[axis] = resultDimSize;

    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType(), dynDims);

    auto toOpFoldResult = [](Value v) -> OpFoldResult {
      auto op = v.getDefiningOp<arith::ConstantIndexOp>();
      if (!op)
        return v;
      return op.getValue();
    };
    Value result = emptyTensor;
    for (auto arg : adaptor.getOperands()) {
      sizes[axis] = rewriter.createOrFold<tensor::DimOp>(loc, arg, axisValue);
      result = rewriter.createOrFold<tensor::InsertSliceOp>(
          loc, arg, result,
          llvm::to_vector(llvm::map_range(offsets, toOpFoldResult)),
          llvm::to_vector(llvm::map_range(sizes, toOpFoldResult)),
          llvm::to_vector(llvm::map_range(strides, toOpFoldResult)));
      offsets[axis] =
          rewriter.createOrFold<arith::AddIOp>(loc, offsets[axis], sizes[axis]);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ReverseConverter : public OpRewritePattern<tosa::ReverseOp> {
public:
  using OpRewritePattern<tosa::ReverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ReverseOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value input = op.getInput();
    auto inputTy = input.getType().template cast<ShapedType>();
    auto resultTy = op.getType().template cast<ShapedType>();
    auto axis = op.getAxis();

    SmallVector<Value> dynDims;
    for (int i = 0; i < inputTy.getRank(); i++) {
      if (inputTy.isDynamicDim(i)) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
      }
    }

    Value axisDimSize = rewriter.create<tensor::DimOp>(loc, input, axis);

    // First fill the output buffer with the init value.
    auto emptyTensor = rewriter
                           .create<tensor::EmptyOp>(loc, inputTy.getShape(),
                                                    inputTy.getElementType(),
                                                    ArrayRef<Value>({dynDims}))
                           .getResult();
    SmallVector<AffineMap, 2> affineMaps = {
        rewriter.getMultiDimIdentityMap(resultTy.getRank())};

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, resultTy, ArrayRef<Value>({}), ValueRange{emptyTensor}, affineMaps,
        getNParallelLoopsAttrs(resultTy.getRank()),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          llvm::SmallVector<Value> indices;
          for (unsigned int i = 0; i < inputTy.getRank(); i++) {
            auto index =
                rewriter.create<linalg::IndexOp>(nestedLoc, i).getResult();
            if (i == axis) {
              auto one = rewriter.create<arith::ConstantIndexOp>(nestedLoc, 1);
              auto sizeMinusOne =
                  rewriter.create<arith::SubIOp>(nestedLoc, axisDimSize, one);
              index = rewriter.create<arith::SubIOp>(nestedLoc, sizeMinusOne,
                                                     index);
            }

            indices.push_back(index);
          }

          auto extract = nestedBuilder.create<tensor::ExtractOp>(
              nestedLoc, input, indices);
          nestedBuilder.create<linalg::YieldOp>(op.getLoc(),
                                                extract.getResult());
        });
    return success();
  }
};

// This converter translate a tile operation to a reshape, broadcast, reshape.
// The first reshape minimally expands each tiled dimension to include a
// proceding size-1 dim. This dim is then broadcasted to the appropriate
// multiple.
struct TileConverter : public OpConversionPattern<tosa::TileOp> {
  using OpConversionPattern<tosa::TileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::TileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput1();
    auto inputTy = input.getType().cast<ShapedType>();
    auto inputShape = inputTy.getShape();
    auto resultTy = op.getType().cast<ShapedType>();
    auto elementTy = inputTy.getElementType();
    int64_t rank = inputTy.getRank();

    SmallVector<int64_t> multiples;
    getValuesFromIntArrayAttribute(op.getMultiples(), multiples);

    // Broadcast the newly added dimensions to their appropriate multiple.
    SmallVector<int64_t, 2> genericShape;
    for (int i = 0; i < rank; i++) {
      int64_t dim = multiples[i];
      genericShape.push_back(dim == -1 ? ShapedType::kDynamic : dim);
      genericShape.push_back(inputShape[i]);
    }

    SmallVector<Value> dynDims;
    for (int i = 0; i < inputTy.getRank(); i++) {
      if (inputTy.isDynamicDim(i) || multiples[i] == -1) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
      }
    }

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), genericShape, elementTy, dynDims);

    // We needs to map the input shape to the non-broadcasted dimensions.
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
      dimExprs.push_back(rewriter.getAffineDimExpr(i * 2 + 1));

    auto readAffineMap =
        AffineMap::get(/*dimCount=*/rank * 2, /*symbolCount=*/0, dimExprs,
                       rewriter.getContext());

    SmallVector<AffineMap, 2> affineMaps = {
        readAffineMap, rewriter.getMultiDimIdentityMap(genericShape.size())};

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, RankedTensorType::get(genericShape, elementTy), input,
        ValueRange{emptyTensor}, affineMaps,
        getNParallelLoopsAttrs(genericShape.size()),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(op.getLoc(), *args.begin());
        });

    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
        op, resultTy, genericOp.getResult(0),
        rewriter.getI64ArrayAttr(resultTy.getShape()));
    return success();
  }
};

class PadConverter : public OpRewritePattern<tosa::PadOp> {
public:
  using OpRewritePattern<tosa::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::PadOp padOp,
                                PatternRewriter &rewriter) const final {
    auto loc = padOp.getLoc();
    auto input = padOp.getInput1();
    auto padding = padOp.getPadding();

    ShapedType inputTy = input.getType().cast<ShapedType>();
    Type elementTy = inputTy.getElementType();
    int64_t rank = inputTy.getRank();

    // Setup the default constantAttr.

    Value padConstant;

    if (padOp.getPadConst()) {
      padConstant = rewriter.createOrFold<tensor::ExtractOp>(
          loc, padOp.getPadConst(), ValueRange({}));
    } else {
      Attribute constantAttr;
      if (elementTy.isa<FloatType>()) {
        constantAttr = rewriter.getFloatAttr(elementTy, 0.0);
      } else if (elementTy.isa<IntegerType>() && !padOp.getQuantizationInfo()) {
        constantAttr = rewriter.getIntegerAttr(elementTy, 0);
      } else if (elementTy.isa<IntegerType>() && padOp.getQuantizationInfo()) {
        int64_t value = padOp.getQuantizationInfo()->getInputZp();
        constantAttr = rewriter.getIntegerAttr(elementTy, value);
      }
      if (constantAttr)
        padConstant = rewriter.create<arith::ConstantOp>(loc, constantAttr);
    }

    if (!padConstant) {
      return rewriter.notifyMatchFailure(
          padOp, "tosa.pad was unable to determine the pad constant value.");
    }

    Value lowIndex =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    Value highIndex =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    SmallVector<OpFoldResult, 3> lowValues;
    SmallVector<OpFoldResult, 3> highValues;

    lowValues.reserve(rank);
    highValues.reserve(rank);

    for (int i = 0; i < rank; i++) {
      Value inputIndex = rewriter.createOrFold<arith::ConstantIndexOp>(loc, i);
      Value lowVal = rewriter.createOrFold<tensor::ExtractOp>(
          loc, padding, ValueRange({inputIndex, lowIndex}));
      Value highVal = rewriter.createOrFold<tensor::ExtractOp>(
          loc, padding, ValueRange({inputIndex, highIndex}));

      lowVal = rewriter.createOrFold<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), lowVal);
      highVal = rewriter.createOrFold<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), highVal);

      lowValues.push_back(lowVal);
      highValues.push_back(highVal);
    }

    auto newPadOp = rewriter.create<tensor::PadOp>(
        loc, padOp.getType(), input, lowValues, highValues, padConstant);

    rewriter.replaceOp(padOp, newPadOp.getResult());
    return success();
  }
};

// Tosa argmax lowering represents the ArgMax op as an linalg.indexed_generic
// op, producing two output buffers.
//
// The first output buffer contains the index of the found maximum value. It is
// initialized to 0 and is resulting integer type.
//
// The second output buffer contains the maximum value found. It is initialized
// to the minimum representable value of the input element type. After being
// populated by indexed_generic, this buffer is disgarded as only the index is
// requested.
//
// The indexed_generic op updates both the maximum value and index if the
// current value exceeds the running max.
class ArgMaxConverter : public OpRewritePattern<tosa::ArgMaxOp> {
public:
  using OpRewritePattern<tosa::ArgMaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ArgMaxOp argmaxOp,
                                PatternRewriter &rewriter) const final {
    auto loc = argmaxOp.getLoc();
    Value input = argmaxOp.getInput();
    auto inputTy = input.getType().cast<ShapedType>();
    auto resultTy = argmaxOp.getOutput().getType().cast<ShapedType>();
    auto inElementTy = inputTy.getElementType();
    auto outElementTy = resultTy.getElementType();
    int axis = argmaxOp.getAxis();
    auto resultMaxTy = RankedTensorType::get(resultTy.getShape(), inElementTy);

    if (!outElementTy.isa<IntegerType>())
      return rewriter.notifyMatchFailure(
          argmaxOp,
          "tosa.arg_max to linalg.* requires integer-like result type");

    SmallVector<Value> dynDims;
    for (int i = 0; i < inputTy.getRank(); i++) {
      if (inputTy.isDynamicDim(i) && i != axis) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
      }
    }

    // First fill the output buffer for the index.
    auto emptyTensorIdx = rewriter
                              .create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                       outElementTy, dynDims)
                              .getResult();
    auto fillValueIdx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(outElementTy, 0));
    auto filledTensorIdx =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{fillValueIdx},
                                    ValueRange{emptyTensorIdx})
            .result();

    // Second fill the output buffer for the running max.
    auto emptyTensorMax = rewriter
                              .create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                       inElementTy, dynDims)
                              .getResult();
    auto fillValueMaxAttr =
        createInitialValueForReduceOp(argmaxOp, inElementTy, rewriter);

    if (!fillValueMaxAttr)
      return rewriter.notifyMatchFailure(
          argmaxOp, "unsupported tosa.argmax element type");

    auto fillValueMax =
        rewriter.create<arith::ConstantOp>(loc, fillValueMaxAttr);
    auto filledTensorMax =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{fillValueMax},
                                    ValueRange{emptyTensorMax})
            .result();

    // We need to reduce along the arg-max axis, with parallel operations along
    // the rest.
    SmallVector<utils::IteratorType, 4> iteratorTypes;
    iteratorTypes.resize(inputTy.getRank(), utils::IteratorType::parallel);
    iteratorTypes[axis] = utils::IteratorType::reduction;

    SmallVector<AffineExpr, 2> srcExprs;
    SmallVector<AffineExpr, 2> dstExprs;
    for (int i = 0, rank = inputTy.getRank(); i != rank; ++i) {
      srcExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
      if (axis != i)
        dstExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
    }

    bool didEncounterError = false;
    auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs, dstExprs});
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, ArrayRef<Type>({resultTy, resultMaxTy}), input,
        ValueRange({filledTensorIdx, filledTensorMax}), maps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          auto newValue = blockArgs[0];
          auto oldIndex = blockArgs[1];
          auto oldValue = blockArgs[2];

          Value newIndex = rewriter.create<arith::IndexCastOp>(
              nestedLoc, oldIndex.getType(),
              rewriter.create<linalg::IndexOp>(loc, axis));

          Value predicate;
          if (inElementTy.isa<FloatType>()) {
            predicate = rewriter.create<arith::CmpFOp>(
                nestedLoc, arith::CmpFPredicate::OGT, newValue, oldValue);
          } else if (inElementTy.isa<IntegerType>()) {
            predicate = rewriter.create<arith::CmpIOp>(
                nestedLoc, arith::CmpIPredicate::sgt, newValue, oldValue);
          } else {
            didEncounterError = true;
            return;
          }

          auto resultMax = rewriter.create<arith::SelectOp>(
              nestedLoc, predicate, newValue, oldValue);
          auto resultIndex = rewriter.create<arith::SelectOp>(
              nestedLoc, predicate, newIndex, oldIndex);
          nestedBuilder.create<linalg::YieldOp>(
              nestedLoc, ValueRange({resultIndex, resultMax}));
        });

    if (didEncounterError)
      return rewriter.notifyMatchFailure(
          argmaxOp, "unsupported tosa.argmax element type");

    rewriter.replaceOp(argmaxOp, linalgOp.getResult(0));
    return success();
  }
};

class GatherConverter : public OpConversionPattern<tosa::GatherOp> {
public:
  using OpConversionPattern<tosa::GatherOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto input = adaptor.getOperands()[0];
    auto indices = adaptor.getOperands()[1];

    auto resultTy = op.getType().cast<ShapedType>();

    auto dynamicDimsOr = checkHasDynamicBatchDims(
        rewriter, op, {input, indices, op.getOutput()});
    if (!dynamicDimsOr.has_value())
      return failure();
    SmallVector<Value> dynamicDims = dynamicDimsOr.value();

    auto resultElementTy = resultTy.getElementType();

    auto loc = op.getLoc();

    auto emptyTensor =
        rewriter
            .create<tensor::EmptyOp>(loc, resultTy.getShape(), resultElementTy,
                                     dynamicDims)
            .getResult();

    SmallVector<AffineMap, 2> affineMaps = {
        AffineMap::get(
            /*dimCount=*/resultTy.getRank(), /*symbolCount=*/0,
            {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)},
            rewriter.getContext()),
        rewriter.getMultiDimIdentityMap(resultTy.getRank())};

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, ArrayRef<Type>({resultTy}), ValueRange{indices},
        ValueRange{emptyTensor}, affineMaps,
        getNParallelLoopsAttrs(resultTy.getRank()),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto indexValue = args[0];
          auto index0 = rewriter.create<linalg::IndexOp>(loc, 0);
          Value index1 = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getIndexType(), indexValue);
          auto index2 = rewriter.create<linalg::IndexOp>(loc, 2);
          Value extract = rewriter.create<tensor::ExtractOp>(
              loc, input, ValueRange{index0, index1, index2});
          rewriter.create<linalg::YieldOp>(loc, extract);
        });
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Lowerings the TableOp to a series of gathers and numerica operations. This
// includes interpolation between the high/low values. For the I8 varient, this
// simplifies to a single gather operation.
class TableConverter : public OpRewritePattern<tosa::TableOp> {
public:
  using OpRewritePattern<tosa::TableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::TableOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value input = op.getInput();
    Value table = op.getTable();
    auto inputTy = input.getType().cast<ShapedType>();
    auto tableTy = table.getType().cast<ShapedType>();
    auto resultTy = op.getType().cast<ShapedType>();

    auto inputElementTy = inputTy.getElementType();
    auto tableElementTy = tableTy.getElementType();
    auto resultElementTy = resultTy.getElementType();

    SmallVector<Value> dynDims;
    for (int i = 0; i < resultTy.getRank(); ++i) {
      if (inputTy.isDynamicDim(i)) {
        dynDims.push_back(
            rewriter.create<tensor::DimOp>(loc, op.getOperand(0), i));
      }
    }

    auto emptyTensor = rewriter
                           .create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                    resultElementTy, dynDims)
                           .getResult();

    SmallVector<AffineMap, 2> affineMaps = {
        rewriter.getMultiDimIdentityMap(resultTy.getRank()),
        rewriter.getMultiDimIdentityMap(resultTy.getRank())};

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultTy, ValueRange({input}), ValueRange{emptyTensor}, affineMaps,
        getNParallelLoopsAttrs(resultTy.getRank()));
    rewriter.replaceOp(op, genericOp.getResult(0));

    {
      OpBuilder::InsertionGuard regionGuard(rewriter);
      Block *block = rewriter.createBlock(
          &genericOp.getRegion(), genericOp.getRegion().end(),
          TypeRange({inputElementTy, resultElementTy}), {loc, loc});

      auto inputValue = block->getArgument(0);
      rewriter.setInsertionPointToStart(block);
      if (inputElementTy.isInteger(8) && tableElementTy.isInteger(8) &&
          resultElementTy.isInteger(8)) {
        Value index = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), inputValue);
        Value offset = rewriter.create<arith::ConstantIndexOp>(loc, 128);
        index = rewriter.create<arith::AddIOp>(loc, rewriter.getIndexType(),
                                               index, offset);
        Value extract =
            rewriter.create<tensor::ExtractOp>(loc, table, ValueRange{index});
        rewriter.create<linalg::YieldOp>(loc, extract);
        return success();
      }

      if (inputElementTy.isInteger(16) && tableElementTy.isInteger(16) &&
          resultElementTy.isInteger(32)) {
        Value extend = rewriter.create<arith::ExtSIOp>(
            loc, rewriter.getI32Type(), inputValue);

        auto offset = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI32IntegerAttr(32768));
        auto seven = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI32IntegerAttr(7));
        auto one = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI32IntegerAttr(1));
        auto b1111111 = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI32IntegerAttr(127));

        // Compute the index and fractional part from the input value:
        // value = value + 32768
        // index = value >> 7;
        // fraction = 0x01111111 & value
        auto extendAdd = rewriter.create<arith::AddIOp>(loc, extend, offset);
        Value index = rewriter.create<arith::ShRUIOp>(loc, extendAdd, seven);
        Value fraction =
            rewriter.create<arith::AndIOp>(loc, extendAdd, b1111111);

        // Extract the base and next values from the table.
        // base = (int32_t) table[index];
        // next = (int32_t) table[index + 1];
        Value indexPlusOne = rewriter.create<arith::AddIOp>(loc, index, one);

        index = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), index);
        indexPlusOne = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), indexPlusOne);

        Value base =
            rewriter.create<tensor::ExtractOp>(loc, table, ValueRange{index});
        Value next = rewriter.create<tensor::ExtractOp>(
            loc, table, ValueRange{indexPlusOne});

        base =
            rewriter.create<arith::ExtSIOp>(loc, rewriter.getI32Type(), base);
        next =
            rewriter.create<arith::ExtSIOp>(loc, rewriter.getI32Type(), next);

        // Use the fractional part to interpolate between the input values:
        // result = (base << 7) + (next - base) * fraction
        Value baseScaled = rewriter.create<arith::ShLIOp>(loc, base, seven);
        Value diff = rewriter.create<arith::SubIOp>(loc, next, base);
        Value diffScaled = rewriter.create<arith::MulIOp>(loc, diff, fraction);
        Value result =
            rewriter.create<arith::AddIOp>(loc, baseScaled, diffScaled);

        rewriter.create<linalg::YieldOp>(loc, result);

        return success();
      }
    }

    return rewriter.notifyMatchFailure(
        op, "unable to create body for tosa.table op");
  }
};

} // namespace

void mlir::tosa::populateTosaToLinalgConversionPatterns(
    RewritePatternSet *patterns) {

  // We have multiple resize coverters to handle degenerate cases.
  patterns->add<GenericResizeConverter>(patterns->getContext(),
                                        /*benefit=*/100);
  patterns->add<BroadcastResizeConverter>(patterns->getContext(),
                                          /*benefit=*/200);

  patterns->add<
      // clang-format off
      PointwiseConverter<tosa::AddOp>,
      PointwiseConverter<tosa::SubOp>,
      PointwiseConverter<tosa::MulOp>,
      PointwiseConverter<tosa::DivOp>,
      PointwiseConverter<tosa::NegateOp>,
      PointwiseConverter<tosa::PowOp>,
      PointwiseConverter<tosa::ReciprocalOp>,
      PointwiseConverter<tosa::RsqrtOp>,
      PointwiseConverter<tosa::LogOp>,
      PointwiseConverter<tosa::ExpOp>,
      PointwiseConverter<tosa::AbsOp>,
      PointwiseConverter<tosa::TanhOp>,
      PointwiseConverter<tosa::BitwiseAndOp>,
      PointwiseConverter<tosa::BitwiseOrOp>,
      PointwiseConverter<tosa::BitwiseNotOp>,
      PointwiseConverter<tosa::BitwiseXorOp>,
      PointwiseConverter<tosa::LogicalAndOp>,
      PointwiseConverter<tosa::LogicalNotOp>,
      PointwiseConverter<tosa::LogicalOrOp>,
      PointwiseConverter<tosa::LogicalXorOp>,
      PointwiseConverter<tosa::CastOp>,
      PointwiseConverter<tosa::LogicalLeftShiftOp>,
      PointwiseConverter<tosa::LogicalRightShiftOp>,
      PointwiseConverter<tosa::ArithmeticRightShiftOp>,
      PointwiseConverter<tosa::ClzOp>,
      PointwiseConverter<tosa::SelectOp>,
      PointwiseConverter<tosa::GreaterOp>,
      PointwiseConverter<tosa::GreaterEqualOp>,
      PointwiseConverter<tosa::EqualOp>,
      PointwiseConverter<tosa::MaximumOp>,
      PointwiseConverter<tosa::MinimumOp>,
      PointwiseConverter<tosa::CeilOp>,
      PointwiseConverter<tosa::FloorOp>,
      PointwiseConverter<tosa::ClampOp>,
      PointwiseConverter<tosa::SigmoidOp>,
      IdentityNConverter<tosa::IdentityOp>,
      ReduceConverter<tosa::ReduceAllOp>,
      ReduceConverter<tosa::ReduceAnyOp>,
      ReduceConverter<tosa::ReduceMinOp>,
      ReduceConverter<tosa::ReduceMaxOp>,
      ReduceConverter<tosa::ReduceSumOp>,
      ReduceConverter<tosa::ReduceProdOp>,
      ArgMaxConverter,
      ConcatConverter,
      GatherConverter,
      PadConverter,
      ReshapeConverterCollapse,
      ReshapeConverterExpand,
      ReshapeConverterCollapseExpand,
      RescaleConverter,
      ReverseConverter,
      TableConverter,
      TileConverter,
      TransposeConverter>(patterns->getContext());
  // clang-format on
}
