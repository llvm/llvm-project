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
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

#include <numeric>

using namespace mlir;
using namespace mlir::tosa;

template <typename T>
static arith::ConstantOp
createConstFromIntAttribute(Operation *op, const std::string &attrName,
                            Type requiredAttrType, OpBuilder &rewriter) {
  auto castedN = static_cast<T>(
      cast<IntegerAttr>(op->getAttr(attrName)).getValue().getSExtValue());
  return rewriter.create<arith::ConstantOp>(
      op->getLoc(), IntegerAttr::get(requiredAttrType, castedN));
}

static Value createLinalgBodyCalculationForElementwiseOp(
    Operation *op, ValueRange args, ArrayRef<Type> resultTypes,
    ConversionPatternRewriter &rewriter) {
  Location loc = op->getLoc();
  auto elementTy =
      cast<ShapedType>(op->getOperand(0).getType()).getElementType();

  // tosa::AbsOp
  if (isa<tosa::AbsOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<math::AbsFOp>(loc, resultTypes, args);

  if (isa<tosa::AbsOp>(op) && isa<IntegerType>(elementTy)) {
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementTy));
    auto neg = rewriter.create<arith::SubIOp>(loc, zero, args[0]);
    return rewriter.create<arith::MaxSIOp>(loc, args[0], neg);
  }

  // tosa::AddOp
  if (isa<tosa::AddOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<arith::AddFOp>(loc, resultTypes, args);

  if (isa<tosa::AddOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.create<arith::AddIOp>(loc, resultTypes, args);

  // tosa::SubOp
  if (isa<tosa::SubOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<arith::SubFOp>(loc, resultTypes, args);

  if (isa<tosa::SubOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.create<arith::SubIOp>(loc, resultTypes, args);

  // tosa::IntDivOp
  if (isa<tosa::IntDivOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.create<arith::DivSIOp>(loc, resultTypes, args);

  // tosa::ReciprocalOp
  if (isa<tosa::ReciprocalOp>(op) && isa<FloatType>(elementTy)) {
    auto one =
        rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(elementTy, 1));
    return rewriter.create<arith::DivFOp>(loc, resultTypes, one, args[0]);
  }

  // tosa::MulOp
  if (isa<tosa::MulOp>(op)) {
    auto shift_val = cast<tosa::MulOp>(op).getShift();

    if (isa<FloatType>(elementTy)) {
      return rewriter.create<arith::MulFOp>(loc, resultTypes, args[0], args[1]);
    }

    if (isa<IntegerType>(elementTy)) {
      int32_t shift = 0;
      ElementsAttr shift_elem;
      if (shift_val.getImpl() &&
          matchPattern(shift_val, m_Constant(&shift_elem))) {
        // Explicit shift is set.
        shift = shift_elem.getValues<IntegerAttr>()[0].getInt();
      }

      Value a = args[0];
      Value b = args[1];
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
  }

  // tosa::NegateOp
  if (isa<tosa::NegateOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<arith::NegFOp>(loc, resultTypes, args);

  if (isa<tosa::NegateOp>(op) && isa<IntegerType>(elementTy)) {
    int64_t inZp = 0, outZp = 0;

    if (cast<tosa::NegateOp>(op).getQuantizationInfo()) {
      auto quantizationInfo = cast<tosa::NegateOp>(op).getQuantizationInfo();
      inZp = quantizationInfo.value().getInputZp();
      outZp = quantizationInfo.value().getOutputZp();
    }

    int32_t inputBitWidth = elementTy.getIntOrFloatBitWidth();
    if (!inZp && !outZp) {
      auto constant = rewriter.create<arith::ConstantOp>(
          loc, IntegerAttr::get(elementTy, 0));
      return rewriter.create<arith::SubIOp>(loc, resultTypes, constant,
                                            args[0]);
    }

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
    Value min = rewriter.create<arith::ConstantIntOp>(
        loc, APInt::getSignedMinValue(inputBitWidth).getSExtValue(),
        intermediateType);
    Value max = rewriter.create<arith::ConstantIntOp>(
        loc, APInt::getSignedMaxValue(inputBitWidth).getSExtValue(),
        intermediateType);
    auto clamp =
        clampIntHelper(loc, sub, min, max, rewriter, /*isUnsigned=*/false);

    // Truncate to the final value.
    return rewriter.create<arith::TruncIOp>(loc, elementTy, clamp);
  }

  // tosa::BitwiseAndOp
  if (isa<tosa::BitwiseAndOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.create<arith::AndIOp>(loc, resultTypes, args);

  // tosa::BitwiseOrOp
  if (isa<tosa::BitwiseOrOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.create<arith::OrIOp>(loc, resultTypes, args);

  // tosa::BitwiseNotOp
  if (isa<tosa::BitwiseNotOp>(op) && isa<IntegerType>(elementTy)) {
    auto allOnesAttr = rewriter.getIntegerAttr(
        elementTy, APInt::getAllOnes(elementTy.getIntOrFloatBitWidth()));
    auto allOnes = rewriter.create<arith::ConstantOp>(loc, allOnesAttr);
    return rewriter.create<arith::XOrIOp>(loc, resultTypes, args[0], allOnes);
  }

  // tosa::BitwiseXOrOp
  if (isa<tosa::BitwiseXorOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.create<arith::XOrIOp>(loc, resultTypes, args);

  // tosa::LogicalLeftShiftOp
  if (isa<tosa::LogicalLeftShiftOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.create<arith::ShLIOp>(loc, resultTypes, args);

  // tosa::LogicalRightShiftOp
  if (isa<tosa::LogicalRightShiftOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.create<arith::ShRUIOp>(loc, resultTypes, args);

  // tosa::ArithmeticRightShiftOp
  if (isa<tosa::ArithmeticRightShiftOp>(op) && isa<IntegerType>(elementTy)) {
    auto result = rewriter.create<arith::ShRSIOp>(loc, resultTypes, args);
    auto round = cast<BoolAttr>(op->getAttr("round")).getValue();
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
  if (isa<tosa::ClzOp>(op) && isa<IntegerType>(elementTy)) {
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
  if (isa<tosa::PowOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<mlir::math::PowFOp>(loc, resultTypes, args);

  // tosa::RsqrtOp
  if (isa<tosa::RsqrtOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<mlir::math::RsqrtOp>(loc, resultTypes, args);

  // tosa::LogOp
  if (isa<tosa::LogOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<mlir::math::LogOp>(loc, resultTypes, args);

  // tosa::ExpOp
  if (isa<tosa::ExpOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<mlir::math::ExpOp>(loc, resultTypes, args);

  // tosa::SinOp
  if (isa<tosa::SinOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<mlir::math::SinOp>(loc, resultTypes, args);

  // tosa::CosOp
  if (isa<tosa::CosOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<mlir::math::CosOp>(loc, resultTypes, args);

  // tosa::TanhOp
  if (isa<tosa::TanhOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<mlir::math::TanhOp>(loc, resultTypes, args);

  // tosa::ErfOp
  if (isa<tosa::ErfOp>(op) && llvm::isa<FloatType>(elementTy))
    return rewriter.create<mlir::math::ErfOp>(loc, resultTypes, args);

  // tosa::GreaterOp
  if (isa<tosa::GreaterOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
                                          args[0], args[1]);

  if (isa<tosa::GreaterOp>(op) && elementTy.isSignlessInteger())
    return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                          args[0], args[1]);

  // tosa::GreaterEqualOp
  if (isa<tosa::GreaterEqualOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE,
                                          args[0], args[1]);

  if (isa<tosa::GreaterEqualOp>(op) && elementTy.isSignlessInteger())
    return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                          args[0], args[1]);

  // tosa::EqualOp
  if (isa<tosa::EqualOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                          args[0], args[1]);

  if (isa<tosa::EqualOp>(op) && elementTy.isSignlessInteger())
    return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                          args[0], args[1]);

  // tosa::SelectOp
  if (isa<tosa::SelectOp>(op)) {
    elementTy = cast<ShapedType>(op->getOperand(1).getType()).getElementType();
    if (isa<FloatType>(elementTy) || isa<IntegerType>(elementTy))
      return rewriter.create<arith::SelectOp>(loc, args[0], args[1], args[2]);
  }

  // tosa::MaximumOp
  if (isa<tosa::MaximumOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::MaximumFOp>(loc, args[0], args[1]);
  }

  if (isa<tosa::MaximumOp>(op) && elementTy.isSignlessInteger()) {
    return rewriter.create<arith::MaxSIOp>(loc, args[0], args[1]);
  }

  // tosa::MinimumOp
  if (isa<tosa::MinimumOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::MinimumFOp>(loc, args[0], args[1]);
  }

  if (isa<tosa::MinimumOp>(op) && elementTy.isSignlessInteger()) {
    return rewriter.create<arith::MinSIOp>(loc, args[0], args[1]);
  }

  // tosa::CeilOp
  if (isa<tosa::CeilOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<math::CeilOp>(loc, resultTypes, args);

  // tosa::FloorOp
  if (isa<tosa::FloorOp>(op) && isa<FloatType>(elementTy))
    return rewriter.create<math::FloorOp>(loc, resultTypes, args);

  // tosa::ClampOp
  if (isa<tosa::ClampOp>(op) && isa<FloatType>(elementTy)) {
    bool losesInfo = false;
    APFloat minApf = cast<FloatAttr>(op->getAttr("min_fp")).getValue();
    APFloat maxApf = cast<FloatAttr>(op->getAttr("max_fp")).getValue();
    minApf.convert(cast<FloatType>(elementTy).getFloatSemantics(),
                   APFloat::rmNearestTiesToEven, &losesInfo);
    maxApf.convert(cast<FloatType>(elementTy).getFloatSemantics(),
                   APFloat::rmNearestTiesToEven, &losesInfo);
    auto min = rewriter.create<arith::ConstantOp>(
        loc, elementTy, rewriter.getFloatAttr(elementTy, minApf));
    auto max = rewriter.create<arith::ConstantOp>(
        loc, elementTy, rewriter.getFloatAttr(elementTy, maxApf));
    return clampFloatHelper(loc, args[0], min, max, rewriter);
  }

  if (isa<tosa::ClampOp>(op) && isa<IntegerType>(elementTy)) {
    auto intTy = cast<IntegerType>(elementTy);
    int64_t min =
        cast<IntegerAttr>(op->getAttr("min_int")).getValue().getSExtValue();
    int64_t max =
        cast<IntegerAttr>(op->getAttr("max_int")).getValue().getSExtValue();

    int64_t minRepresentable = std::numeric_limits<int64_t>::min();
    int64_t maxRepresentable = std::numeric_limits<int64_t>::max();
    if (intTy.isUnsignedInteger()) {
      minRepresentable = 0;
      if (intTy.getIntOrFloatBitWidth() <= 63) {
        maxRepresentable =
            (int64_t)APInt::getMaxValue(intTy.getIntOrFloatBitWidth())
                .getZExtValue();
      }
    } else if (intTy.getIntOrFloatBitWidth() <= 64) {
      // Ensure that min & max fit into signed n-bit constants.
      minRepresentable = APInt::getSignedMinValue(intTy.getIntOrFloatBitWidth())
                             .getSExtValue();
      maxRepresentable = APInt::getSignedMaxValue(intTy.getIntOrFloatBitWidth())
                             .getSExtValue();
    }
    // Ensure that the bounds are representable as n-bit signed/unsigned
    // integers.
    min = std::max(min, minRepresentable);
    max = std::max(max, minRepresentable);
    min = std::min(min, maxRepresentable);
    max = std::min(max, maxRepresentable);

    auto minVal = rewriter.create<arith::ConstantIntOp>(
        loc, min, intTy.getIntOrFloatBitWidth());
    auto maxVal = rewriter.create<arith::ConstantIntOp>(
        loc, max, intTy.getIntOrFloatBitWidth());
    return clampIntHelper(loc, args[0], minVal, maxVal, rewriter,
                          intTy.isUnsignedInteger());
  }

  // tosa::SigmoidOp
  if (isa<tosa::SigmoidOp>(op) && isa<FloatType>(elementTy)) {
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

    if (isa<FloatType>(srcTy) && isa<FloatType>(dstTy) && bitExtend)
      return rewriter.create<arith::ExtFOp>(loc, resultTypes, args,
                                            std::nullopt);

    if (isa<FloatType>(srcTy) && isa<FloatType>(dstTy) && !bitExtend)
      return rewriter.create<arith::TruncFOp>(loc, resultTypes, args,
                                              std::nullopt);

    // 1-bit integers need to be treated as signless.
    if (srcTy.isInteger(1) && arith::UIToFPOp::areCastCompatible(srcTy, dstTy))
      return rewriter.create<arith::UIToFPOp>(loc, resultTypes, args,
                                              std::nullopt);

    if (srcTy.isInteger(1) && isa<IntegerType>(dstTy) && bitExtend)
      return rewriter.create<arith::ExtUIOp>(loc, resultTypes, args,
                                             std::nullopt);

    // Unsigned integers need an unrealized cast so that they can be passed
    // to UIToFP.
    if (srcTy.isUnsignedInteger() && isa<FloatType>(dstTy)) {
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
    if (isa<FloatType>(srcTy) && dstTy.isInteger(1)) {
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(srcTy, 0.0));
      return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE,
                                            args.front(), zero);
    }

    if (arith::FPToSIOp::areCastCompatible(srcTy, dstTy)) {
      auto rounded = rewriter.create<math::RoundEvenOp>(loc, args[0]);

      const auto &fltSemantics = cast<FloatType>(srcTy).getFloatSemantics();
      // Check whether neither int min nor int max can be represented in the
      // input floating-point type due to too short exponent range.
      if (static_cast<int>(dstTy.getIntOrFloatBitWidth()) - 1 >
          APFloat::semanticsMaxExponent(fltSemantics)) {
        // Use cmp + select to replace infinites by int min / int max. Other
        // integral values can be represented in the integer space.
        auto conv = rewriter.create<arith::FPToSIOp>(loc, dstTy, rounded);
        auto posInf = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(getElementTypeOrSelf(srcTy),
                                       APFloat::getInf(fltSemantics)));
        auto negInf = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(
                     getElementTypeOrSelf(srcTy),
                     APFloat::getInf(fltSemantics, /*Negative=*/true)));
        auto overflow = rewriter.create<arith::CmpFOp>(
            loc, arith::CmpFPredicate::UEQ, rounded, posInf);
        auto underflow = rewriter.create<arith::CmpFOp>(
            loc, arith::CmpFPredicate::UEQ, rounded, negInf);
        auto intMin = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(
                     getElementTypeOrSelf(dstTy),
                     APInt::getSignedMinValue(dstTy.getIntOrFloatBitWidth())));
        auto intMax = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(
                     getElementTypeOrSelf(dstTy),
                     APInt::getSignedMaxValue(dstTy.getIntOrFloatBitWidth())));
        auto maxClamped =
            rewriter.create<arith::SelectOp>(loc, overflow, intMax, conv);
        return rewriter.create<arith::SelectOp>(loc, underflow, intMin,
                                                maxClamped);
      }

      auto intMinFP = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(
                   getElementTypeOrSelf(srcTy),
                   APInt::getSignedMinValue(dstTy.getIntOrFloatBitWidth())
                       .getSExtValue()));

      // Check whether the mantissa has enough bits to represent int max.
      if (cast<FloatType>(srcTy).getFPMantissaWidth() >=
          dstTy.getIntOrFloatBitWidth() - 1) {
        // Int min can also be represented since it is a power of two and thus
        // consists of a single leading bit. Therefore we can clamp the input
        // in the floating-point domain.

        auto intMaxFP = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(
                     getElementTypeOrSelf(srcTy),
                     APInt::getSignedMaxValue(dstTy.getIntOrFloatBitWidth())
                         .getSExtValue()));

        Value clamped =
            clampFloatHelper(loc, rounded, intMinFP, intMaxFP, rewriter);
        return rewriter.create<arith::FPToSIOp>(loc, dstTy, clamped);
      }

      // Due to earlier check we know exponant range is big enough to represent
      // int min. We can therefore rely on int max + 1 being representable as
      // well because it's just int min with a positive sign. So clamp the min
      // value and compare against that to select the max int value if needed.
      auto intMaxPlusOneFP = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(
                   getElementTypeOrSelf(srcTy),
                   static_cast<double>(
                       APInt::getSignedMaxValue(dstTy.getIntOrFloatBitWidth())
                           .getSExtValue()) +
                       1.0f));

      auto intMax = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(
                   getElementTypeOrSelf(dstTy),
                   APInt::getSignedMaxValue(dstTy.getIntOrFloatBitWidth())));
      auto minClampedFP =
          rewriter.create<arith::MaximumFOp>(loc, rounded, intMinFP);
      auto minClamped =
          rewriter.create<arith::FPToSIOp>(loc, dstTy, minClampedFP);
      auto overflow = rewriter.create<arith::CmpFOp>(
          loc, arith::CmpFPredicate::UGE, rounded, intMaxPlusOneFP);
      return rewriter.create<arith::SelectOp>(loc, overflow, intMax,
                                              minClamped);
    }

    // Casting to boolean, integers need to only be checked as not-equal to
    // zero.
    if (isa<IntegerType>(srcTy) && dstTy.isInteger(1)) {
      Value zero = rewriter.create<arith::ConstantIntOp>(
          loc, 0, srcTy.getIntOrFloatBitWidth());
      return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                            args.front(), zero);
    }

    if (isa<IntegerType>(srcTy) && isa<IntegerType>(dstTy) && bitExtend)
      return rewriter.create<arith::ExtSIOp>(loc, resultTypes, args,
                                             std::nullopt);

    if (isa<IntegerType>(srcTy) && isa<IntegerType>(dstTy) && !bitExtend) {
      return rewriter.create<arith::TruncIOp>(loc, dstTy, args[0]);
    }
  }

  (void)rewriter.notifyMatchFailure(
      op, "unhandled op for linalg body calculation for elementwise op");
  return nullptr;
}

static Value expandRank(PatternRewriter &rewriter, Location loc, Value tensor,
                        int64_t rank) {
  // No need to expand if we are already at the desired rank
  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  assert(tensorType && "expected a ranked tensor type");
  int64_t tensorRank = tensorType.getRank();
  int64_t numExtraDims = rank - tensorRank;
  assert(numExtraDims >= 0 && "cannot expand tensor to a lower rank");
  if (!numExtraDims)
    return tensor;

  // Compute reassociation indices
  SmallVector<ReassociationIndices> reassociationIndices(tensorRank);
  int64_t index = 0;
  if (tensorRank != 0) {
    for (index = 0; index <= numExtraDims; index++)
      reassociationIndices[0].push_back(index);
    for (size_t position = 1; position < reassociationIndices.size();
         position++)
      reassociationIndices[position].push_back(index++);
  }

  // Compute result type
  SmallVector<int64_t> resultShape;
  for (index = 0; index < numExtraDims; index++)
    resultShape.push_back(1);
  for (auto size : tensorType.getShape())
    resultShape.push_back(size);
  auto resultType =
      RankedTensorType::get(resultShape, tensorType.getElementType());

  // Emit 'tensor.expand_shape' op
  return rewriter.create<tensor::ExpandShapeOp>(loc, resultType, tensor,
                                                reassociationIndices);
}

static SmallVector<Value> expandInputRanks(PatternRewriter &rewriter,
                                           Location loc, ValueRange operands,
                                           int64_t rank) {
  return llvm::map_to_vector(operands, [&](Value operand) {
    return expandRank(rewriter, loc, operand, rank);
  });
}

using IndexPool = DenseMap<int64_t, Value>;

// Emit an 'arith.constant' op for the given index if it has not been created
// yet, or return an existing constant. This will prevent an excessive creation
// of redundant constants, easing readability of emitted code for unit tests.
static Value createIndex(PatternRewriter &rewriter, Location loc,
                         IndexPool &indexPool, int64_t index) {
  auto [it, inserted] = indexPool.try_emplace(index);
  if (inserted)
    it->second =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(index));
  return it->second;
}

static Value getTensorDim(PatternRewriter &rewriter, Location loc,
                          IndexPool &indexPool, Value tensor, int64_t index) {
  auto indexValue = createIndex(rewriter, loc, indexPool, index);
  return rewriter.create<tensor::DimOp>(loc, tensor, indexValue).getResult();
}

static OpFoldResult getOrFoldTensorDim(PatternRewriter &rewriter, Location loc,
                                       IndexPool &indexPool, Value tensor,
                                       int64_t index) {
  auto shapedType = dyn_cast<ShapedType>(tensor.getType());
  assert(shapedType && shapedType.hasRank() && "expected a ranked shaped type");
  assert(index >= 0 && index < shapedType.getRank() && "index out of bounds");
  if (shapedType.isDynamicDim(index))
    return getTensorDim(rewriter, loc, indexPool, tensor, index);
  return rewriter.getIndexAttr(shapedType.getDimSize(index));
}

static bool operandsAndResultsRanked(Operation *operation) {
  auto isRanked = [](Value value) {
    return isa<RankedTensorType>(value.getType());
  };
  return llvm::all_of(operation->getOperands(), isRanked) &&
         llvm::all_of(operation->getResults(), isRanked);
}

// Compute the runtime dimension size for dimension 'dim' of the output by
// inspecting input 'operands', all of which are expected to have the same rank.
// This function returns a pair {targetSize, masterOperand}.
//
// The runtime size of the output dimension is returned either as a statically
// computed attribute or as a runtime SSA value.
//
// If the target size was inferred directly from one dominating operand, that
// operand is returned in 'masterOperand'. If the target size is inferred from
// multiple operands, 'masterOperand' is set to nullptr.
static std::pair<OpFoldResult, Value>
computeTargetSize(PatternRewriter &rewriter, Location loc, IndexPool &indexPool,
                  ValueRange operands, int64_t dim) {
  // If any input operand contains a static size greater than 1 for this
  // dimension, that is the target size. An occurrence of an additional static
  // dimension greater than 1 with a different value is undefined behavior.
  for (auto operand : operands) {
    auto size = cast<RankedTensorType>(operand.getType()).getDimSize(dim);
    if (!ShapedType::isDynamic(size) && size > 1)
      return {rewriter.getIndexAttr(size), operand};
  }

  // Filter operands with dynamic dimension
  auto operandsWithDynamicDim =
      llvm::filter_to_vector(operands, [&](Value operand) {
        return cast<RankedTensorType>(operand.getType()).isDynamicDim(dim);
      });

  // If no operand has a dynamic dimension, it means all sizes were 1
  if (operandsWithDynamicDim.empty())
    return {rewriter.getIndexAttr(1), operands.front()};

  // Emit code that computes the runtime size for this dimension. If there is
  // only one operand with a dynamic dimension, it is considered the master
  // operand that determines the runtime size of the output dimension.
  auto targetSize =
      getTensorDim(rewriter, loc, indexPool, operandsWithDynamicDim[0], dim);
  if (operandsWithDynamicDim.size() == 1)
    return {targetSize, operandsWithDynamicDim[0]};

  // Calculate maximum size among all dynamic dimensions
  for (size_t i = 1; i < operandsWithDynamicDim.size(); i++) {
    auto nextSize =
        getTensorDim(rewriter, loc, indexPool, operandsWithDynamicDim[i], dim);
    targetSize = rewriter.create<arith::MaxUIOp>(loc, targetSize, nextSize);
  }
  return {targetSize, nullptr};
}

// Compute the runtime output size for all dimensions. This function returns
// a pair {targetShape, masterOperands}.
static std::pair<SmallVector<OpFoldResult>, SmallVector<Value>>
computeTargetShape(PatternRewriter &rewriter, Location loc,
                   IndexPool &indexPool, ValueRange operands) {
  assert(!operands.empty());
  auto rank = cast<RankedTensorType>(operands.front().getType()).getRank();
  SmallVector<OpFoldResult> targetShape;
  SmallVector<Value> masterOperands;
  for (auto dim : llvm::seq<int64_t>(0, rank)) {
    auto [targetSize, masterOperand] =
        computeTargetSize(rewriter, loc, indexPool, operands, dim);
    targetShape.push_back(targetSize);
    masterOperands.push_back(masterOperand);
  }
  return {targetShape, masterOperands};
}

static Value broadcastDynamicDimension(PatternRewriter &rewriter, Location loc,
                                       IndexPool &indexPool, Value operand,
                                       int64_t dim, OpFoldResult targetSize,
                                       Value masterOperand) {
  // Nothing to do if this is a static dimension
  auto rankedTensorType = cast<RankedTensorType>(operand.getType());
  if (!rankedTensorType.isDynamicDim(dim))
    return operand;

  // If the target size for this dimension was directly inferred by only taking
  // this operand into account, there is no need to broadcast. This is an
  // optimization that will prevent redundant control flow, and constitutes the
  // main motivation for tracking "master operands".
  if (operand == masterOperand)
    return operand;

  // Affine maps for 'linalg.generic' op
  auto rank = rankedTensorType.getRank();
  SmallVector<AffineExpr> affineExprs;
  for (auto index : llvm::seq<int64_t>(0, rank)) {
    auto affineExpr = index == dim ? rewriter.getAffineConstantExpr(0)
                                   : rewriter.getAffineDimExpr(index);
    affineExprs.push_back(affineExpr);
  }
  auto broadcastAffineMap =
      AffineMap::get(rank, 0, affineExprs, rewriter.getContext());
  auto identityAffineMap = rewriter.getMultiDimIdentityMap(rank);
  SmallVector<AffineMap> affineMaps = {broadcastAffineMap, identityAffineMap};

  // Check if broadcast is necessary
  auto one = createIndex(rewriter, loc, indexPool, 1);
  auto runtimeSize = getTensorDim(rewriter, loc, indexPool, operand, dim);
  auto broadcastNecessary = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, runtimeSize, one);

  // Emit 'then' region of 'scf.if'
  auto emitThenRegion = [&](OpBuilder &opBuilder, Location loc) {
    // It is not safe to cache constants across regions.
    // New constants could potentially violate dominance requirements.
    IndexPool localPool;

    // Emit 'tensor.empty' op
    SmallVector<OpFoldResult> outputTensorShape;
    for (auto index : llvm::seq<int64_t>(0, rank)) {
      auto size = index == dim ? targetSize
                               : getOrFoldTensorDim(rewriter, loc, localPool,
                                                    operand, index);
      outputTensorShape.push_back(size);
    }
    Value outputTensor = opBuilder.create<tensor::EmptyOp>(
        loc, outputTensorShape, rankedTensorType.getElementType());

    // Emit 'linalg.generic' op
    auto resultTensor =
        opBuilder
            .create<linalg::GenericOp>(
                loc, outputTensor.getType(), operand, outputTensor, affineMaps,
                getNParallelLoopsAttrs(rank),
                [&](OpBuilder &opBuilder, Location loc, ValueRange blockArgs) {
                  // Emit 'linalg.yield' op
                  opBuilder.create<linalg::YieldOp>(loc, blockArgs.front());
                })
            .getResult(0);

    // Cast to original operand type if necessary
    auto castResultTensor = rewriter.createOrFold<tensor::CastOp>(
        loc, operand.getType(), resultTensor);

    // Emit 'scf.yield' op
    opBuilder.create<scf::YieldOp>(loc, castResultTensor);
  };

  // Emit 'else' region of 'scf.if'
  auto emitElseRegion = [&](OpBuilder &opBuilder, Location loc) {
    opBuilder.create<scf::YieldOp>(loc, operand);
  };

  // Emit 'scf.if' op
  auto ifOp = rewriter.create<scf::IfOp>(loc, broadcastNecessary,
                                         emitThenRegion, emitElseRegion);
  return ifOp.getResult(0);
}

static Value broadcastDynamicDimensions(PatternRewriter &rewriter, Location loc,
                                        IndexPool &indexPool, Value operand,
                                        ArrayRef<OpFoldResult> targetShape,
                                        ArrayRef<Value> masterOperands) {
  int64_t rank = cast<RankedTensorType>(operand.getType()).getRank();
  assert((int64_t)targetShape.size() == rank);
  assert((int64_t)masterOperands.size() == rank);
  for (auto index : llvm::seq<int64_t>(0, rank))
    operand =
        broadcastDynamicDimension(rewriter, loc, indexPool, operand, index,
                                  targetShape[index], masterOperands[index]);
  return operand;
}

static SmallVector<Value>
broadcastDynamicDimensions(PatternRewriter &rewriter, Location loc,
                           IndexPool &indexPool, ValueRange operands,
                           ArrayRef<OpFoldResult> targetShape,
                           ArrayRef<Value> masterOperands) {
  // No need to broadcast for unary operations
  if (operands.size() == 1)
    return operands;

  // Broadcast dynamic dimensions operand by operand
  return llvm::map_to_vector(operands, [&](Value operand) {
    return broadcastDynamicDimensions(rewriter, loc, indexPool, operand,
                                      targetShape, masterOperands);
  });
}

static LogicalResult
emitElementwiseComputation(ConversionPatternRewriter &rewriter, Location loc,
                           Operation *operation, ValueRange operands,
                           ArrayRef<OpFoldResult> targetShape,
                           const TypeConverter &converter) {
  // Generate output tensor
  auto resultType = cast_or_null<RankedTensorType>(
      converter.convertType(operation->getResultTypes().front()));
  if (!resultType) {
    return rewriter.notifyMatchFailure(operation, "failed to convert type");
  }
  Value outputTensor = rewriter.create<tensor::EmptyOp>(
      loc, targetShape, resultType.getElementType());

  // Create affine maps. Input affine maps broadcast static dimensions of size
  // 1. The output affine map is an identity map.
  //
  auto rank = resultType.getRank();
  auto affineMaps = llvm::map_to_vector(operands, [&](Value operand) {
    auto shape = cast<ShapedType>(operand.getType()).getShape();
    SmallVector<AffineExpr> affineExprs;
    for (auto it : llvm::enumerate(shape)) {
      // Prefer producting identity maps whenever possible (i.e. no broadcasting
      // needed) because some transforms (like reshape folding)
      // do not support affine constant exprs.
      bool requiresBroadcast =
          (it.value() == 1 && resultType.getDimSize(it.index()) != 1);
      auto affineExpr = requiresBroadcast
                            ? rewriter.getAffineConstantExpr(0)
                            : rewriter.getAffineDimExpr(it.index());
      affineExprs.push_back(affineExpr);
    }
    return AffineMap::get(rank, 0, affineExprs, rewriter.getContext());
  });
  affineMaps.push_back(rewriter.getMultiDimIdentityMap(rank));

  // Emit 'linalg.generic' op
  bool encounteredError = false;
  auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc, outputTensor.getType(), operands, outputTensor, affineMaps,
      getNParallelLoopsAttrs(rank),
      [&](OpBuilder &opBuilder, Location loc, ValueRange blockArgs) {
        Value opResult = createLinalgBodyCalculationForElementwiseOp(
            operation, blockArgs.take_front(operation->getNumOperands()),
            {resultType.getElementType()}, rewriter);
        if (!opResult) {
          encounteredError = true;
          return;
        }
        opBuilder.create<linalg::YieldOp>(loc, opResult);
      });
  if (encounteredError)
    return rewriter.notifyMatchFailure(
        operation, "unable to create linalg.generic body for elementwise op");

  // Cast 'linalg.generic' result into original result type if needed
  auto castResult = rewriter.createOrFold<tensor::CastOp>(
      loc, resultType, linalgOp->getResult(0));
  rewriter.replaceOp(operation, castResult);
  return success();
}

static LogicalResult
elementwiseMatchAndRewriteHelper(Operation *operation, ValueRange operands,
                                 ConversionPatternRewriter &rewriter,
                                 const TypeConverter &converter) {

  // Collect op properties
  assert(operation->getNumResults() == 1 && "elementwise op expects 1 result");
  assert(operation->getNumOperands() >= 1 &&
         "elementwise op expects at least 1 operand");
  if (!operandsAndResultsRanked(operation))
    return rewriter.notifyMatchFailure(operation,
                                       "Unranked tensors not supported");

  // Lower operation
  IndexPool indexPool;
  auto loc = operation->getLoc();
  auto rank =
      cast<RankedTensorType>(operation->getResultTypes().front()).getRank();
  // For the mul op we need to avoid expanding the rank of the optional shift
  // input.
  auto operandsToExpand =
      isa<tosa::MulOp>(operation) ? operands.take_front(2) : operands;

  auto expandedOperands =
      expandInputRanks(rewriter, loc, operandsToExpand, rank);
  auto [targetShape, masterOperands] =
      computeTargetShape(rewriter, loc, indexPool, expandedOperands);
  auto broadcastOperands = broadcastDynamicDimensions(
      rewriter, loc, indexPool, expandedOperands, targetShape, masterOperands);
  return emitElementwiseComputation(rewriter, loc, operation, broadcastOperands,
                                    targetShape, converter);
}

// Returns the constant initial value for a given reduction operation. The
// attribute type varies depending on the element type required.
static TypedAttr createInitialValueForReduceOp(Operation *op, Type elementTy,
                                               PatternRewriter &rewriter) {
  if (isa<tosa::ReduceSumOp>(op) && isa<FloatType>(elementTy))
    return rewriter.getFloatAttr(elementTy, 0.0);

  if (isa<tosa::ReduceSumOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.getIntegerAttr(elementTy, 0);

  if (isa<tosa::ReduceProdOp>(op) && isa<FloatType>(elementTy))
    return rewriter.getFloatAttr(elementTy, 1.0);

  if (isa<tosa::ReduceProdOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.getIntegerAttr(elementTy, 1);

  if (isa<tosa::ReduceMinOp>(op) && isa<FloatType>(elementTy))
    return rewriter.getFloatAttr(
        elementTy, APFloat::getLargest(
                       cast<FloatType>(elementTy).getFloatSemantics(), false));

  if (isa<tosa::ReduceMinOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.getIntegerAttr(
        elementTy, APInt::getSignedMaxValue(elementTy.getIntOrFloatBitWidth()));

  if (isa<tosa::ReduceMaxOp>(op) && isa<FloatType>(elementTy))
    return rewriter.getFloatAttr(
        elementTy, APFloat::getLargest(
                       cast<FloatType>(elementTy).getFloatSemantics(), true));

  if (isa<tosa::ReduceMaxOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.getIntegerAttr(
        elementTy, APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth()));

  if (isa<tosa::ReduceAllOp>(op) && elementTy.isInteger(1))
    return rewriter.getIntegerAttr(elementTy, APInt::getAllOnes(1));

  if (isa<tosa::ReduceAnyOp>(op) && elementTy.isInteger(1))
    return rewriter.getIntegerAttr(elementTy, APInt::getZero(1));

  if (isa<tosa::ArgMaxOp>(op) && isa<FloatType>(elementTy))
    return rewriter.getFloatAttr(
        elementTy, APFloat::getLargest(
                       cast<FloatType>(elementTy).getFloatSemantics(), true));

  if (isa<tosa::ArgMaxOp>(op) && isa<IntegerType>(elementTy))
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
  if (isa<tosa::ReduceSumOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::AddFOp>(loc, args);
  }

  if (isa<tosa::ReduceSumOp>(op) && isa<IntegerType>(elementTy)) {
    return rewriter.create<arith::AddIOp>(loc, args);
  }

  if (isa<tosa::ReduceProdOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::MulFOp>(loc, args);
  }

  if (isa<tosa::ReduceProdOp>(op) && isa<IntegerType>(elementTy)) {
    return rewriter.create<arith::MulIOp>(loc, args);
  }

  if (isa<tosa::ReduceMinOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::MinimumFOp>(loc, args[0], args[1]);
  }

  if (isa<tosa::ReduceMinOp>(op) && isa<IntegerType>(elementTy)) {
    return rewriter.create<arith::MinSIOp>(loc, args[0], args[1]);
  }

  if (isa<tosa::ReduceMaxOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::MaximumFOp>(loc, args[0], args[1]);
  }

  if (isa<tosa::ReduceMaxOp>(op) && isa<IntegerType>(elementTy)) {
    return rewriter.create<arith::MaxSIOp>(loc, args[0], args[1]);
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
  auto inputTy = cast<ShapedType>(op->getOperand(0).getType());
  auto resultTy = cast<ShapedType>(op->getResult(0).getType());
  auto elementTy = resultTy.getElementType();
  Value input = op->getOperand(0);

  SmallVector<int64_t> reduceShape;
  SmallVector<Value> dynDims;
  for (unsigned i = 0; i < inputTy.getRank(); i++) {
    if (axis != i) {
      reduceShape.push_back(inputTy.getDimSize(i));
      if (inputTy.isDynamicDim(i))
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
    }
  }

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

  bool didEncounterError = false;
  auto linalgOp = rewriter.create<linalg::ReduceOp>(
      loc, input, filledTensor, axis,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto result = createLinalgBodyCalculationForReduceOp(
            op, blockArgs, elementTy, rewriter);
        if (result)
          didEncounterError = true;

        nestedBuilder.create<linalg::YieldOp>(loc, result);
      });

  if (!didEncounterError)
    return rewriter.notifyMatchFailure(
        op, "unable to create linalg.generic body for reduce op");

  SmallVector<ReassociationExprs, 4> reassociationMap;
  uint64_t expandInputRank =
      cast<ShapedType>(linalgOp.getResults()[0].getType()).getRank();
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

  // Lower directly to `tensor::ExpandShapeOp` instead of `tosa::ReshapeOp`,
  // since here we know which dimension to expand, and `tosa::ReshapeOp` would
  // not have access to such information. This matters when handling dynamically
  // sized tensors.
  rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
      op, resultTy, linalgOp.getResults()[0], reassociationMap);
  return success();
}

namespace {

template <typename SrcOp>
class PointwiseConverter : public OpConversionPattern<SrcOp> {
public:
  using OpConversionPattern<SrcOp>::OpConversionPattern;
  using typename OpConversionPattern<SrcOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const final {
    return elementwiseMatchAndRewriteHelper(
        op, operands.getOperands(), rewriter, *this->getTypeConverter());
  }
};

class RescaleConverter : public OpRewritePattern<tosa::RescaleOp> {
public:
  using OpRewritePattern<tosa::RescaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::RescaleOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto inputTy = cast<ShapedType>(op.getInput().getType());
    auto outputTy = cast<ShapedType>(op.getOutput().getType());
    unsigned rank = inputTy.getRank();

    // This is an illegal configuration. terminate and log an error
    if (op.getDoubleRound() && !op.getScale32())
      return rewriter.notifyMatchFailure(
          op, "tosa.rescale requires scale32 for double_round to be true");

    if (!isa<IntegerType>(inputTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "only support integer type");

    SmallVector<Value> dynDims;
    for (int i = 0; i < outputTy.getRank(); i++) {
      if (outputTy.isDynamicDim(i)) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
      }
    }

    // The shift and multiplier values.
    SmallVector<int32_t> multiplierValues(op.getMultiplier());
    SmallVector<int8_t> shiftValues(op.getShift());

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
            if (op.getInputUnsigned()) {
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
              cast<IntegerType>(blockArgs.back().getType());
          unsigned outBitWidth = outIntType.getWidth();

          int32_t intMin = APInt::getSignedMinValue(outBitWidth).getSExtValue();
          int32_t intMax = APInt::getSignedMaxValue(outBitWidth).getSExtValue();

          // Unsigned integers have a difference output value.
          if (op.getOutputUnsigned()) {
            intMin = 0;
            intMax = APInt::getMaxValue(outBitWidth).getZExtValue();
          }

          auto intMinVal = nestedBuilder.create<arith::ConstantOp>(
              loc, nestedBuilder.getI32IntegerAttr(intMin));
          auto intMaxVal = nestedBuilder.create<arith::ConstantOp>(
              loc, nestedBuilder.getI32IntegerAttr(intMax));

          value = clampIntHelper(nestedLoc, value, intMinVal, intMaxVal,
                                 nestedBuilder, /*isUnsigned=*/false);

          if (outIntType.getWidth() < 32) {
            value = nestedBuilder.create<arith::TruncIOp>(
                nestedLoc, rewriter.getIntegerType(outIntType.getWidth()),
                value);
          }

          nestedBuilder.create<linalg::YieldOp>(loc, value);
        });

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

// Handle the resize case where the input is a 1x1 image. This case
// can entirely avoiding having extract operations which target much
// more difficult to optimize away.
class ResizeUnaryConverter : public OpRewritePattern<tosa::ResizeOp> {
public:
  using OpRewritePattern<tosa::ResizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ResizeOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder builder(loc, rewriter);
    auto input = op.getInput();
    auto inputTy = cast<RankedTensorType>(input.getType());
    auto resultTy = cast<RankedTensorType>(op.getType());
    const bool isBilinear = op.getMode() == "BILINEAR";

    auto inputH = inputTy.getDimSize(1);
    auto inputW = inputTy.getDimSize(2);
    auto outputH = resultTy.getDimSize(1);
    auto outputW = resultTy.getDimSize(2);

    if (inputH != 1 || inputW != 1 || outputH != 1 || outputW != 1)
      return rewriter.notifyMatchFailure(
          op, "tosa.resize is not a pure 1x1->1x1 image operation");

    // TODO(suderman): These string values should be declared the TOSA dialect.
    if (op.getMode() != "NEAREST_NEIGHBOR" && op.getMode() != "BILINEAR")
      return rewriter.notifyMatchFailure(
          op, "tosa.resize mode should be NEAREST_NEIGHBOR or BILINEAR");

    if (inputTy == resultTy) {
      rewriter.replaceOp(op, input);
      return success();
    }

    SmallVector<int64_t> scale;
    if (!tosa::getConstShapeValue(op.getScale().getDefiningOp(), scale)) {
      return failure();
    }

    // Collapse the unit width and height away.
    SmallVector<ReassociationExprs, 4> reassociationMap(2);
    reassociationMap[0].push_back(builder.getAffineDimExpr(0));
    reassociationMap[1].push_back(builder.getAffineDimExpr(1));
    reassociationMap[1].push_back(builder.getAffineDimExpr(2));
    reassociationMap[1].push_back(builder.getAffineDimExpr(3));

    auto collapseTy =
        RankedTensorType::get({inputTy.getDimSize(0), inputTy.getDimSize(3)},
                              inputTy.getElementType());
    Value collapse = builder.create<tensor::CollapseShapeOp>(collapseTy, input,
                                                             reassociationMap);

    // Get any dynamic shapes that appear in the input format.
    llvm::SmallVector<Value> outputDynSize;
    if (inputTy.isDynamicDim(0))
      outputDynSize.push_back(builder.create<tensor::DimOp>(input, 0));
    if (inputTy.isDynamicDim(3))
      outputDynSize.push_back(builder.create<tensor::DimOp>(input, 3));

    // Generate the elementwise operation for casting scaling the input value.
    auto genericTy = collapseTy.clone(resultTy.getElementType());
    Value empty = builder.create<tensor::EmptyOp>(
        genericTy.getShape(), resultTy.getElementType(), outputDynSize);
    auto genericMap = rewriter.getMultiDimIdentityMap(genericTy.getRank());
    SmallVector<utils::IteratorType> iterators(genericTy.getRank(),
                                               utils::IteratorType::parallel);

    auto generic = builder.create<linalg::GenericOp>(
        genericTy, ValueRange{collapse}, ValueRange{empty},
        ArrayRef<AffineMap>{genericMap, genericMap}, iterators,
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

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, resultTy, generic.getResults()[0], reassociationMap);
    return success();
  }
};

class GenericResizeConverter : public OpRewritePattern<tosa::ResizeOp> {
public:
  using OpRewritePattern<tosa::ResizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ResizeOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    auto input = op.getInput();
    auto inputTy = cast<ShapedType>(input.getType());
    auto resultTy = cast<ShapedType>(op.getType());
    auto resultETy = resultTy.getElementType();

    bool floatingPointMode = resultETy.isF16() || resultETy.isF32();
    auto floatTy = resultETy.isF16() ? b.getF16Type() : b.getF32Type();

    auto imageH = inputTy.getShape()[1];
    auto imageW = inputTy.getShape()[2];

    auto dynamicDimsOr =
        checkHasDynamicBatchDims(rewriter, op, {input, op.getOutput()});
    if (!dynamicDimsOr.has_value())
      return rewriter.notifyMatchFailure(
          op, "unable to get dynamic dimensions of tosa.resize");

    if (op.getMode() != "NEAREST_NEIGHBOR" && op.getMode() != "BILINEAR")
      return rewriter.notifyMatchFailure(
          op, "tosa.resize mode should be NEAREST_NEIGHBOR or BILINEAR");

    SmallVector<AffineMap, 2> affineMaps = {
        rewriter.getMultiDimIdentityMap(resultTy.getRank())};
    auto emptyTensor = b.create<tensor::EmptyOp>(resultTy.getShape(), resultETy,
                                                 *dynamicDimsOr);
    auto genericOp = b.create<linalg::GenericOp>(
        resultTy, ValueRange({}), ValueRange{emptyTensor}, affineMaps,
        getNParallelLoopsAttrs(resultTy.getRank()));
    Value resize = genericOp.getResult(0);

    {
      OpBuilder::InsertionGuard regionGuard(b);
      b.createBlock(&genericOp.getRegion(), genericOp.getRegion().end(),
                    TypeRange({resultETy}), loc);
      Value batch = b.create<linalg::IndexOp>(0);
      Value y = b.create<linalg::IndexOp>(1);
      Value x = b.create<linalg::IndexOp>(2);
      Value channel = b.create<linalg::IndexOp>(3);

      Value zeroI32 =
          b.create<arith::ConstantOp>(b.getZeroAttr(b.getI32Type()));
      Value zeroFp = b.create<arith::ConstantOp>(b.getZeroAttr(floatTy));
      Value hMax = b.create<arith::ConstantOp>(b.getI32IntegerAttr(imageH - 1));
      Value wMax = b.create<arith::ConstantOp>(b.getI32IntegerAttr(imageW - 1));

      Value inY = b.create<arith::IndexCastOp>(b.getI32Type(), y);
      Value inX = b.create<arith::IndexCastOp>(b.getI32Type(), x);

      SmallVector<int64_t> scale, offset, border;
      if (!tosa::getConstShapeValue(op.getScale().getDefiningOp(), scale) ||
          !tosa::getConstShapeValue(op.getOffset().getDefiningOp(), offset) ||
          !tosa::getConstShapeValue(op.getBorder().getDefiningOp(), border)) {
        return rewriter.notifyMatchFailure(
            op, "tosa.resize scale/offset/border should have compile time "
                "constant values.");
      }

      Value yScaleN, yScaleD, xScaleN, xScaleD;
      yScaleN = b.create<arith::ConstantOp>(b.getI32IntegerAttr(scale[0]));
      yScaleD = b.create<arith::ConstantOp>(b.getI32IntegerAttr(scale[1]));
      xScaleN = b.create<arith::ConstantOp>(b.getI32IntegerAttr(scale[2]));
      xScaleD = b.create<arith::ConstantOp>(b.getI32IntegerAttr(scale[3]));

      Value yOffset, xOffset, yBorder, xBorder;
      yOffset = b.create<arith::ConstantOp>(b.getI32IntegerAttr(offset[0]));
      xOffset = b.create<arith::ConstantOp>(b.getI32IntegerAttr(offset[1]));
      yBorder = b.create<arith::ConstantOp>(b.getI32IntegerAttr(border[0]));
      xBorder = b.create<arith::ConstantOp>(b.getI32IntegerAttr(border[1]));

      // Compute the ix and dx values for both the X and Y dimensions.
      auto getIndexAndDeltaFp = [&](Value &index, Value &delta, Value in,
                                    Value scaleN, Value scaleD, Value offset,
                                    int size, ImplicitLocOpBuilder &b) {
        if (size == 1) {
          index = zeroI32;
          delta = zeroFp;
          return;
        }
        // x = x * scale_d + offset;
        // ix = floor(x / scale_n)
        Value val = b.create<arith::MulIOp>(in, scaleD);
        val = b.create<arith::AddIOp>(val, offset);
        index = b.create<arith::FloorDivSIOp>(val, scaleN);

        // rx = x % scale_n
        // dx = rx / scale_n
        Value r = b.create<arith::RemSIOp>(val, scaleN);
        Value rFp = b.create<arith::SIToFPOp>(floatTy, r);
        Value scaleNfp = b.create<arith::UIToFPOp>(floatTy, scaleN);
        delta = b.create<arith::DivFOp>(rFp, scaleNfp);
      };

      // Compute the ix and dx values for the X and Y dimensions - int case.
      auto getIndexAndDeltaInt = [&](Value &index, Value &delta, Value in,
                                     Value scaleN, Value scaleD, Value offset,
                                     int size, ImplicitLocOpBuilder &b) {
        if (size == 1) {
          index = zeroI32;
          delta = zeroI32;
          return;
        }
        // x = x * scale_d + offset;
        // ix = floor(x / scale_n)
        //  dx = x - ix * scale_n;
        Value val = b.create<arith::MulIOp>(in, scaleD);
        val = b.create<arith::AddIOp>(val, offset);
        index = b.create<arith::DivSIOp>(val, scaleN);
        delta = b.create<arith::MulIOp>(index, scaleN);
        delta = b.create<arith::SubIOp>(val, delta);
      };

      Value ix, iy, dx, dy;
      if (floatingPointMode) {
        getIndexAndDeltaFp(iy, dy, inY, yScaleN, yScaleD, yOffset, imageH, b);
        getIndexAndDeltaFp(ix, dx, inX, xScaleN, xScaleD, xOffset, imageW, b);
      } else {
        getIndexAndDeltaInt(iy, dy, inY, yScaleN, yScaleD, yOffset, imageH, b);
        getIndexAndDeltaInt(ix, dx, inX, xScaleN, xScaleD, xOffset, imageW, b);
      }

      if (op.getMode() == "NEAREST_NEIGHBOR") {
        auto one = b.create<arith::ConstantOp>(b.getI32IntegerAttr(1));

        auto getNearestIndexAndClamp = [&](Value val, Value dval, Value scale,
                                           Value max, int size,
                                           ImplicitLocOpBuilder &b) -> Value {
          if (size == 1) {
            return b.create<arith::ConstantIndexOp>(0);
          }

          Value pred;
          if (floatingPointMode) {
            auto h = b.create<arith::ConstantOp>(b.getFloatAttr(floatTy, 0.5f));
            pred = b.create<arith::CmpFOp>(arith::CmpFPredicate::OGE, dval, h);
          } else {
            Value dvalDouble = b.create<arith::ShLIOp>(dval, one);
            pred = b.create<arith::CmpIOp>(arith::CmpIPredicate::sge,
                                           dvalDouble, scale);
          }

          auto offset = b.create<arith::SelectOp>(pred, one, zeroI32);
          val = b.create<arith::AddIOp>(val, offset);
          val = clampIntHelper(loc, val, zeroI32, max, b, /*isUnsigned=*/false);
          return b.create<arith::IndexCastOp>(b.getIndexType(), val);
        };

        iy = getNearestIndexAndClamp(iy, dy, yScaleN, hMax, imageH, b);
        ix = getNearestIndexAndClamp(ix, dx, xScaleN, wMax, imageW, b);

        Value result = b.create<tensor::ExtractOp>(
            input, ValueRange{batch, iy, ix, channel});

        b.create<linalg::YieldOp>(result);
      } else {
        // The mode here must be BILINEAR.
        assert(op.getMode() == "BILINEAR");

        auto oneVal = b.create<arith::ConstantOp>(b.getI32IntegerAttr(1));

        auto getClampedIdxs = [&](Value &val0, Value &val1, int size, Value in,
                                  Value max, ImplicitLocOpBuilder &b) {
          val0 = in;
          val1 = b.create<arith::AddIOp>(val0, oneVal);
          val0 =
              clampIntHelper(loc, val0, zeroI32, max, b, /*isUnsigned=*/false);
          val1 =
              clampIntHelper(loc, val1, zeroI32, max, b, /*isUnsigned=*/false);
          val0 = b.create<arith::IndexCastOp>(b.getIndexType(), val0);
          val1 = b.create<arith::IndexCastOp>(b.getIndexType(), val1);
        };

        // Linalg equivalent to the section below:
        //    int16_t iy0 = apply_max(iy, 0);
        //    int16_t iy1 = apply_min(iy + 1, IH - 1);
        //    int16_t ix0 = apply_max(ix, 0);
        //    int16_t ix1 = apply_min(ix + 1, IW - 1);
        Value x0, x1, y0, y1;
        getClampedIdxs(y0, y1, imageH, iy, hMax, b);
        getClampedIdxs(x0, x1, imageW, ix, wMax, b);

        Value y0x0 = b.create<tensor::ExtractOp>(
            input, ValueRange{batch, y0, x0, channel});
        Value y0x1 = b.create<tensor::ExtractOp>(
            input, ValueRange{batch, y0, x1, channel});
        Value y1x0 = b.create<tensor::ExtractOp>(
            input, ValueRange{batch, y1, x0, channel});
        Value y1x1 = b.create<tensor::ExtractOp>(
            input, ValueRange{batch, y1, x1, channel});

        if (floatingPointMode) {
          auto oneVal =
              b.create<arith::ConstantOp>(b.getFloatAttr(floatTy, 1.0f));
          auto interpolate = [&](Value val0, Value val1, Value delta,
                                 int inputSize,
                                 ImplicitLocOpBuilder &b) -> Value {
            if (inputSize == 1)
              return val0;
            Value oneMinusDelta = b.create<arith::SubFOp>(oneVal, delta);
            Value mul0 = b.create<arith::MulFOp>(val0, oneMinusDelta);
            Value mul1 = b.create<arith::MulFOp>(val1, delta);
            return b.create<arith::AddFOp>(mul0, mul1);
          };

          // Linalg equivalent to the section below:
          //   topAcc = v00 * (unit_x - dx);
          //   topAcc += v01 * dx;
          Value topAcc = interpolate(y0x0, y0x1, dx, imageW, b);

          // Linalg equivalent to the section below:
          //   bottomAcc = v10 * (unit_x - dx);
          //   bottomAcc += v11 * dx;
          Value bottomAcc = interpolate(y1x0, y1x1, dx, imageW, b);

          // Linalg equivalent to the section below:
          //   result = topAcc * (unit_y - dy) + bottomAcc * dy
          Value result = interpolate(topAcc, bottomAcc, dy, imageH, b);
          b.create<linalg::YieldOp>(result);
        } else {
          // Perform in quantized space.
          y0x0 = b.create<arith::ExtSIOp>(resultETy, y0x0);
          y0x1 = b.create<arith::ExtSIOp>(resultETy, y0x1);
          y1x0 = b.create<arith::ExtSIOp>(resultETy, y1x0);
          y1x1 = b.create<arith::ExtSIOp>(resultETy, y1x1);

          const int64_t deltaBitwidth = dx.getType().getIntOrFloatBitWidth();
          if (resultETy.getIntOrFloatBitWidth() > deltaBitwidth) {
            dx = b.create<arith::ExtSIOp>(resultETy, dx);
            dy = b.create<arith::ExtSIOp>(resultETy, dy);
          }

          Value yScaleNExt = yScaleN;
          Value xScaleNExt = xScaleN;

          const int64_t scaleBitwidth =
              xScaleN.getType().getIntOrFloatBitWidth();
          if (resultETy.getIntOrFloatBitWidth() > scaleBitwidth) {
            yScaleNExt = b.create<arith::ExtSIOp>(resultETy, yScaleN);
            xScaleNExt = b.create<arith::ExtSIOp>(resultETy, xScaleN);
          }

          auto interpolate = [](Value val0, Value val1, Value weight1,
                                Value scale, int inputSize,
                                ImplicitLocOpBuilder &b) -> Value {
            if (inputSize == 1)
              return b.create<arith::MulIOp>(val0, scale);
            Value weight0 = b.create<arith::SubIOp>(scale, weight1);
            Value mul0 = b.create<arith::MulIOp>(val0, weight0);
            Value mul1 = b.create<arith::MulIOp>(val1, weight1);
            return b.create<arith::AddIOp>(mul0, mul1);
          };

          Value topAcc = interpolate(y0x0, y0x1, dx, xScaleNExt, imageW, b);
          Value bottomAcc = interpolate(y1x0, y1x1, dx, xScaleNExt, imageW, b);
          Value result =
              interpolate(topAcc, bottomAcc, dy, yScaleNExt, imageH, b);
          b.create<linalg::YieldOp>(result);
        }
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

class ReverseConverter : public OpRewritePattern<tosa::ReverseOp> {
public:
  using OpRewritePattern<tosa::ReverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ReverseOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value input = op.getInput1();
    auto inputTy = cast<ShapedType>(input.getType());
    auto resultTy = cast<ShapedType>(op.getType());
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
            Value index =
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
    auto inputTy = cast<ShapedType>(input.getType());
    auto inputShape = inputTy.getShape();
    auto resultTy = cast<ShapedType>(op.getType());
    auto elementTy = inputTy.getElementType();
    int64_t rank = inputTy.getRank();

    SmallVector<int64_t> multiples;
    if (failed(op.getConstantMultiples(multiples)))
      return failure();

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
        rewriter.getDenseI64ArrayAttr(resultTy.getShape()));
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
    auto inputTy = cast<ShapedType>(input.getType());
    auto resultTy = cast<ShapedType>(argmaxOp.getOutput().getType());
    auto inElementTy = inputTy.getElementType();
    auto outElementTy = resultTy.getElementType();
    int axis = argmaxOp.getAxis();
    auto resultMaxTy = RankedTensorType::get(resultTy.getShape(), inElementTy);

    if (!isa<IntegerType>(outElementTy))
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
    auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs, dstExprs},
                                             rewriter.getContext());
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
          if (isa<FloatType>(inElementTy)) {
            predicate = rewriter.create<arith::CmpFOp>(
                nestedLoc, arith::CmpFPredicate::OGT, newValue, oldValue);
          } else if (isa<IntegerType>(inElementTy)) {
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

    auto valuesTy =
        dyn_cast_or_null<RankedTensorType>(op.getValues().getType());
    auto resultTy = cast<ShapedType>(op.getType());

    if (!valuesTy)
      return rewriter.notifyMatchFailure(op, "unranked tensors not supported");

    auto dynamicDims = inferDynamicDimsForGather(
        rewriter, op.getLoc(), adaptor.getValues(), adaptor.getIndices());

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

  static llvm::SmallVector<Value> inferDynamicDimsForGather(OpBuilder &builder,
                                                            Location loc,
                                                            Value values,
                                                            Value indices) {
    llvm::SmallVector<Value> results;

    auto addDynamicDimension = [&](Value source, int64_t dim) {
      auto sz = tensor::getMixedSize(builder, loc, source, dim);
      if (auto dimValue = llvm::dyn_cast_if_present<Value>(sz))
        results.push_back(dimValue);
    };

    addDynamicDimension(values, 0);
    addDynamicDimension(indices, 1);
    addDynamicDimension(values, 2);
    return results;
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
    Value input = op.getInput1();
    Value table = op.getTable();
    auto inputTy = cast<ShapedType>(input.getType());
    auto tableTy = cast<ShapedType>(table.getType());
    auto resultTy = cast<ShapedType>(op.getType());

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

struct RFFT2dConverter final : public OpRewritePattern<RFFT2dOp> {
  using OpRewritePattern<RFFT2dOp>::OpRewritePattern;

  static bool isRankedTensor(Type type) { return isa<RankedTensorType>(type); }

  static OpFoldResult halfPlusOne(OpBuilder &builder, Location loc,
                                  OpFoldResult ofr) {
    auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto two = builder.create<arith::ConstantIndexOp>(loc, 2);

    auto value = getValueOrCreateConstantIndexOp(builder, loc, ofr);
    auto divBy2 = builder.createOrFold<arith::DivUIOp>(loc, value, two);
    auto plusOne = builder.createOrFold<arith::AddIOp>(loc, divBy2, one);
    return getAsOpFoldResult(plusOne);
  }

  static RankedTensorType
  computeOutputShape(OpBuilder &builder, Location loc, Value input,
                     llvm::SmallVectorImpl<Value> &dynamicSizes) {
    // Get [N, H, W]
    auto dims = tensor::getMixedSizes(builder, loc, input);

    // Set W = (W / 2) + 1 to account for the half-sized W dimension of the
    // output tensors.
    dims[2] = halfPlusOne(builder, loc, dims[2]);

    llvm::SmallVector<int64_t, 3> staticSizes;
    dispatchIndexOpFoldResults(dims, dynamicSizes, staticSizes);

    auto elementType = cast<RankedTensorType>(input.getType()).getElementType();
    return RankedTensorType::get(staticSizes, elementType);
  }

  static Value createZeroTensor(PatternRewriter &rewriter, Location loc,
                                RankedTensorType type,
                                llvm::ArrayRef<Value> dynamicSizes) {
    auto emptyTensor =
        rewriter.create<tensor::EmptyOp>(loc, type, dynamicSizes);
    auto fillValueAttr = rewriter.getZeroAttr(type.getElementType());
    auto fillValue = rewriter.create<arith::ConstantOp>(loc, fillValueAttr);
    auto filledTensor = rewriter
                            .create<linalg::FillOp>(loc, ValueRange{fillValue},
                                                    ValueRange{emptyTensor})
                            .result();
    return filledTensor;
  }

  static Value castIndexToFloat(OpBuilder &builder, Location loc,
                                FloatType type, Value value) {
    auto integerVal = builder.create<arith::IndexCastUIOp>(
        loc,
        type.getIntOrFloatBitWidth() > 32 ? builder.getI64Type()
                                          : builder.getI32Type(),
        value);

    return builder.create<arith::UIToFPOp>(loc, type, integerVal);
  }

  static Value createLinalgIndex(OpBuilder &builder, Location loc,
                                 FloatType type, int64_t index) {
    auto indexVal = builder.create<linalg::IndexOp>(loc, index);
    return castIndexToFloat(builder, loc, type, indexVal);
  }

  template <typename... Args>
  static llvm::SmallVector<AffineExpr, 4> affineDimsExpr(OpBuilder &builder,
                                                         Args... args) {
    return {builder.getAffineDimExpr(args)...};
  }

  LogicalResult matchAndRewrite(RFFT2dOp rfft2d,
                                PatternRewriter &rewriter) const override {
    if (!llvm::all_of(rfft2d->getOperandTypes(), isRankedTensor) ||
        !llvm::all_of(rfft2d->getResultTypes(), isRankedTensor)) {
      return rewriter.notifyMatchFailure(rfft2d,
                                         "only supports ranked tensors");
    }

    auto loc = rfft2d.getLoc();
    auto input = rfft2d.getInput();
    auto elementType =
        dyn_cast<FloatType>(cast<ShapedType>(input.getType()).getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(rfft2d,
                                         "only supports float element types");

    // Compute the output type and set of dynamic sizes
    llvm::SmallVector<Value> dynamicSizes;
    auto outputType = computeOutputShape(rewriter, loc, input, dynamicSizes);

    // Iterator types for the linalg.generic implementation
    llvm::SmallVector<utils::IteratorType, 5> iteratorTypes = {
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::parallel, utils::IteratorType::reduction,
        utils::IteratorType::reduction};

    // Inputs/outputs to the linalg.generic implementation
    llvm::SmallVector<Value> genericOpInputs = {input};
    llvm::SmallVector<Value> genericOpOutputs = {
        createZeroTensor(rewriter, loc, outputType, dynamicSizes),
        createZeroTensor(rewriter, loc, outputType, dynamicSizes)};

    // Indexing maps for input and output tensors
    auto indexingMaps = AffineMap::inferFromExprList(
        llvm::ArrayRef{affineDimsExpr(rewriter, 0, 3, 4),
                       affineDimsExpr(rewriter, 0, 1, 2),
                       affineDimsExpr(rewriter, 0, 1, 2)},
        rewriter.getContext());

    // Width and height dimensions of the original input.
    auto dimH = rewriter.createOrFold<tensor::DimOp>(loc, input, 1);
    auto dimW = rewriter.createOrFold<tensor::DimOp>(loc, input, 2);

    // Constants and dimension sizes
    auto twoPiAttr = rewriter.getFloatAttr(elementType, 6.283185307179586);
    auto twoPi = rewriter.create<arith::ConstantOp>(loc, twoPiAttr);
    auto constH = castIndexToFloat(rewriter, loc, elementType, dimH);
    auto constW = castIndexToFloat(rewriter, loc, elementType, dimW);

    auto buildBody = [&](OpBuilder &builder, Location loc, ValueRange args) {
      Value valReal = args[0];
      Value sumReal = args[1];
      Value sumImag = args[2];

      // Indices for angle computation
      Value oy = builder.create<linalg::IndexOp>(loc, 1);
      Value ox = builder.create<linalg::IndexOp>(loc, 2);
      Value iy = builder.create<linalg::IndexOp>(loc, 3);
      Value ix = builder.create<linalg::IndexOp>(loc, 4);

      // Calculating angle without integer parts of components as sin/cos are
      // periodic: angle = 2 * pi() * ( ( (iy * oy) % H) / H + ( (ix * ox) % W )
      // / W);
      auto iyXoy = builder.create<index::MulOp>(loc, iy, oy);
      auto ixXox = builder.create<index::MulOp>(loc, ix, ox);

      auto iyRem = builder.create<index::RemUOp>(loc, iyXoy, dimH);
      auto ixRem = builder.create<index::RemUOp>(loc, ixXox, dimW);

      auto iyRemFloat = castIndexToFloat(builder, loc, elementType, iyRem);
      auto ixRemFloat = castIndexToFloat(builder, loc, elementType, ixRem);

      auto yComponent = builder.create<arith::DivFOp>(loc, iyRemFloat, constH);
      auto xComponent = builder.create<arith::DivFOp>(loc, ixRemFloat, constW);
      auto sumXY = builder.create<arith::AddFOp>(loc, yComponent, xComponent);
      auto angle = builder.create<arith::MulFOp>(loc, twoPi, sumXY);

      // realComponent = valReal * cos(angle)
      // imagComponent = valReal * sin(angle)
      auto cosAngle = builder.create<math::CosOp>(loc, angle);
      auto sinAngle = builder.create<math::SinOp>(loc, angle);
      auto realComponent =
          builder.create<arith::MulFOp>(loc, valReal, cosAngle);
      auto imagComponent =
          builder.create<arith::MulFOp>(loc, valReal, sinAngle);

      // outReal = sumReal + realComponent
      // outImag = sumImag - imagComponent
      auto outReal = builder.create<arith::AddFOp>(loc, sumReal, realComponent);
      auto outImag = builder.create<arith::SubFOp>(loc, sumImag, imagComponent);

      builder.create<linalg::YieldOp>(loc, ValueRange{outReal, outImag});
    };

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        rfft2d, rfft2d.getResultTypes(), genericOpInputs, genericOpOutputs,
        indexingMaps, iteratorTypes, buildBody);

    return success();
  }
};

struct FFT2dConverter final : OpRewritePattern<FFT2dOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FFT2dOp fft2d,
                                PatternRewriter &rewriter) const override {
    if (!llvm::all_of(fft2d->getOperandTypes(),
                      RFFT2dConverter::isRankedTensor) ||
        !llvm::all_of(fft2d->getResultTypes(),
                      RFFT2dConverter::isRankedTensor)) {
      return rewriter.notifyMatchFailure(fft2d, "only supports ranked tensors");
    }

    Location loc = fft2d.getLoc();
    Value input_real = fft2d.getInputReal();
    Value input_imag = fft2d.getInputImag();
    BoolAttr inverse = fft2d.getInverseAttr();

    auto real_el_ty = cast<FloatType>(
        cast<ShapedType>(input_real.getType()).getElementType());
    [[maybe_unused]] auto imag_el_ty = cast<FloatType>(
        cast<ShapedType>(input_imag.getType()).getElementType());

    assert(real_el_ty == imag_el_ty);

    // Compute the output type and set of dynamic sizes
    SmallVector<Value> dynamicSizes;

    // Get [N, H, W]
    auto dims = tensor::getMixedSizes(rewriter, loc, input_real);

    SmallVector<int64_t, 3> staticSizes;
    dispatchIndexOpFoldResults(dims, dynamicSizes, staticSizes);

    auto outputType = RankedTensorType::get(staticSizes, real_el_ty);

    // Iterator types for the linalg.generic implementation
    SmallVector<utils::IteratorType, 5> iteratorTypes = {
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::parallel, utils::IteratorType::reduction,
        utils::IteratorType::reduction};

    // Inputs/outputs to the linalg.generic implementation
    SmallVector<Value> genericOpInputs = {input_real, input_imag};
    SmallVector<Value> genericOpOutputs = {
        RFFT2dConverter::createZeroTensor(rewriter, loc, outputType,
                                          dynamicSizes),
        RFFT2dConverter::createZeroTensor(rewriter, loc, outputType,
                                          dynamicSizes)};

    // Indexing maps for input and output tensors
    auto indexingMaps = AffineMap::inferFromExprList(
        ArrayRef{RFFT2dConverter::affineDimsExpr(rewriter, 0, 3, 4),
                 RFFT2dConverter::affineDimsExpr(rewriter, 0, 3, 4),
                 RFFT2dConverter::affineDimsExpr(rewriter, 0, 1, 2),
                 RFFT2dConverter::affineDimsExpr(rewriter, 0, 1, 2)},
        rewriter.getContext());

    // Width and height dimensions of the original input.
    auto dimH = rewriter.createOrFold<tensor::DimOp>(loc, input_real, 1);
    auto dimW = rewriter.createOrFold<tensor::DimOp>(loc, input_real, 2);

    // Constants and dimension sizes
    auto twoPiAttr = rewriter.getFloatAttr(real_el_ty, 6.283185307179586);
    auto twoPi = rewriter.create<arith::ConstantOp>(loc, twoPiAttr);
    Value constH =
        RFFT2dConverter::castIndexToFloat(rewriter, loc, real_el_ty, dimH);
    Value constW =
        RFFT2dConverter::castIndexToFloat(rewriter, loc, real_el_ty, dimW);

    auto buildBody = [&](OpBuilder &builder, Location loc, ValueRange args) {
      Value valReal = args[0];
      Value valImag = args[1];
      Value sumReal = args[2];
      Value sumImag = args[3];

      // Indices for angle computation
      Value oy = builder.create<linalg::IndexOp>(loc, 1);
      Value ox = builder.create<linalg::IndexOp>(loc, 2);
      Value iy = builder.create<linalg::IndexOp>(loc, 3);
      Value ix = builder.create<linalg::IndexOp>(loc, 4);

      // float_t angle = sign_val * 2 * pi() * ( ( (iy * oy) % H) / H + ( (ix *
      // ox) % W ) / W);
      auto iyXoy = builder.create<index::MulOp>(loc, iy, oy);
      auto ixXox = builder.create<index::MulOp>(loc, ix, ox);

      auto iyRem = builder.create<index::RemUOp>(loc, iyXoy, dimH);
      auto ixRem = builder.create<index::RemUOp>(loc, ixXox, dimW);

      auto iyRemFloat =
          RFFT2dConverter::castIndexToFloat(builder, loc, real_el_ty, iyRem);
      auto ixRemFloat =
          RFFT2dConverter::castIndexToFloat(builder, loc, real_el_ty, ixRem);

      auto yComponent = builder.create<arith::DivFOp>(loc, iyRemFloat, constH);
      auto xComponent = builder.create<arith::DivFOp>(loc, ixRemFloat, constW);

      auto sumXY = builder.create<arith::AddFOp>(loc, yComponent, xComponent);
      auto angle = builder.create<arith::MulFOp>(loc, twoPi, sumXY);

      if (inverse.getValue()) {
        angle = builder.create<arith::MulFOp>(
            loc, angle,
            rewriter.create<arith::ConstantOp>(
                loc, rewriter.getFloatAttr(real_el_ty, -1.0)));
      }

      // realComponent = val_real * cos(a) + val_imag * sin(a);
      // imagComponent = -val_real * sin(a) + val_imag * cos(a);
      auto cosAngle = builder.create<math::CosOp>(loc, angle);
      auto sinAngle = builder.create<math::SinOp>(loc, angle);

      auto rcos = builder.create<arith::MulFOp>(loc, valReal, cosAngle);
      auto rsin = builder.create<arith::MulFOp>(loc, valImag, sinAngle);
      auto realComponent = builder.create<arith::AddFOp>(loc, rcos, rsin);

      auto icos = builder.create<arith::MulFOp>(loc, valImag, cosAngle);
      auto isin = builder.create<arith::MulFOp>(loc, valReal, sinAngle);

      auto imagComponent = builder.create<arith::SubFOp>(loc, icos, isin);

      // outReal = sumReal + realComponent
      // outImag = sumImag - imagComponent
      auto outReal = builder.create<arith::AddFOp>(loc, sumReal, realComponent);
      auto outImag = builder.create<arith::AddFOp>(loc, sumImag, imagComponent);

      builder.create<linalg::YieldOp>(loc, ValueRange{outReal, outImag});
    };

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        fft2d, fft2d.getResultTypes(), genericOpInputs, genericOpOutputs,
        indexingMaps, iteratorTypes, buildBody);

    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaToLinalgConversionPatterns(
    const TypeConverter &converter, RewritePatternSet *patterns) {

  // We have multiple resize coverters to handle degenerate cases.
  patterns->add<GenericResizeConverter>(patterns->getContext(),
                                        /*benefit=*/100);
  patterns->add<ResizeUnaryConverter>(patterns->getContext(),
                                      /*benefit=*/200);

  patterns->add<
      // clang-format off
      PointwiseConverter<tosa::AddOp>,
      PointwiseConverter<tosa::SubOp>,
      PointwiseConverter<tosa::MulOp>,
      PointwiseConverter<tosa::IntDivOp>,
      PointwiseConverter<tosa::NegateOp>,
      PointwiseConverter<tosa::PowOp>,
      PointwiseConverter<tosa::ReciprocalOp>,
      PointwiseConverter<tosa::RsqrtOp>,
      PointwiseConverter<tosa::LogOp>,
      PointwiseConverter<tosa::ExpOp>,
      PointwiseConverter<tosa::AbsOp>,
      PointwiseConverter<tosa::SinOp>,
      PointwiseConverter<tosa::CosOp>,
      PointwiseConverter<tosa::TanhOp>,
      PointwiseConverter<tosa::ErfOp>,
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
      PointwiseConverter<tosa::SigmoidOp>
        >(converter, patterns->getContext());

  patterns->add<
      IdentityNConverter<tosa::IdentityOp>,
      ReduceConverter<tosa::ReduceAllOp>,
      ReduceConverter<tosa::ReduceAnyOp>,
      ReduceConverter<tosa::ReduceMinOp>,
      ReduceConverter<tosa::ReduceMaxOp>,
      ReduceConverter<tosa::ReduceSumOp>,
      ReduceConverter<tosa::ReduceProdOp>,
      ArgMaxConverter,
      GatherConverter,
      RescaleConverter,
      ReverseConverter,
      RFFT2dConverter,
      FFT2dConverter,
      TableConverter,
      TileConverter>(patterns->getContext());
  // clang-format on
}
