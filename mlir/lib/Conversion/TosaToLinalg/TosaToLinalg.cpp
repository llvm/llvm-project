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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

#include <type_traits>

using namespace mlir;
using namespace mlir::tosa;

// Helper function to materialize the semantically correct compare and select
// operations given a binary operation with a specific NaN propagation mode.
//
// In the case of "PROPAGATE" semantics no compare and selection is required and
// this function does nothing.
//
// In the case of "IGNORE" semantics this function materializes a comparison of
// the current operands to the op which will return true for any NaN
// argument and then selects between the non-NaN operation argument and the
// calculated result based on whether the lhs or rhs is NaN or not. In pseudo
// code:
//
// In the case that the op is operating on non floating point types we ignore
// the attribute completely, this is consistent with the TOSA spec which has
// the following wording: "This attribute is ignored by non floating-point
// types."
//
// binary<op>(lhs, rhs):
//   result = op(lhs, rhs)
//   if lhs == NaN return rhs
//   if rhs == NaN return lhs
//   return result
template <typename OpTy>
static Value
materializeBinaryNanCheckIfRequired(OpTy op, PatternRewriter &rewriter,
                                    Value lhs, Value rhs, Value result) {
  // NaN propagation has no meaning for non floating point types.
  if (!isa<FloatType>(getElementTypeOrSelf(lhs)))
    return result;

  auto nanMode = op.getNanMode();
  if (nanMode == NanPropagationMode::PROPAGATE)
    return result;

  // Unordered comparison of NaN against itself will always return true.
  Value lhsIsNaN = arith::CmpFOp::create(rewriter, op.getLoc(),
                                         arith::CmpFPredicate::UNO, lhs, lhs);
  Value rhsIsNaN = arith::CmpFOp::create(rewriter, op.getLoc(),
                                         arith::CmpFPredicate::UNO, rhs, rhs);
  Value rhsOrResult =
      arith::SelectOp::create(rewriter, op.getLoc(), lhsIsNaN, rhs, result);
  return arith::SelectOp::create(rewriter, op.getLoc(), rhsIsNaN, lhs,
                                 rhsOrResult);
}

static Value createLinalgBodyCalculationForElementwiseOp(
    Operation *op, ValueRange args, ArrayRef<Type> resultTypes,
    ConversionPatternRewriter &rewriter) {
  Location loc = op->getLoc();
  auto elementTy =
      cast<ShapedType>(op->getOperand(0).getType()).getElementType();

  // tosa::AbsOp
  if (isa<tosa::AbsOp>(op) && isa<FloatType>(elementTy))
    return math::AbsFOp::create(rewriter, loc, resultTypes, args);

  if (isa<tosa::AbsOp>(op) && isa<IntegerType>(elementTy)) {
    auto zero = arith::ConstantOp::create(rewriter, loc,
                                          rewriter.getZeroAttr(elementTy));
    auto neg = arith::SubIOp::create(rewriter, loc, zero, args[0]);
    return arith::MaxSIOp::create(rewriter, loc, args[0], neg);
  }

  // tosa::AddOp
  if (isa<tosa::AddOp>(op) && isa<FloatType>(elementTy))
    return arith::AddFOp::create(rewriter, loc, resultTypes, args);

  if (isa<tosa::AddOp>(op) && isa<IntegerType>(elementTy))
    return arith::AddIOp::create(rewriter, loc, resultTypes, args);

  // tosa::SubOp
  if (isa<tosa::SubOp>(op) && isa<FloatType>(elementTy))
    return arith::SubFOp::create(rewriter, loc, resultTypes, args);

  if (isa<tosa::SubOp>(op) && isa<IntegerType>(elementTy))
    return arith::SubIOp::create(rewriter, loc, resultTypes, args);

  // tosa::IntDivOp
  if (isa<tosa::IntDivOp>(op) && isa<IntegerType>(elementTy))
    return arith::DivSIOp::create(rewriter, loc, resultTypes, args);

  // tosa::ReciprocalOp
  if (isa<tosa::ReciprocalOp>(op) && isa<FloatType>(elementTy)) {
    auto one =
        arith::ConstantOp::create(rewriter, loc, FloatAttr::get(elementTy, 1));
    return arith::DivFOp::create(rewriter, loc, resultTypes, one, args[0]);
  }

  // tosa::MulOp
  if (isa<tosa::MulOp>(op)) {
    auto shiftVal = cast<tosa::MulOp>(op).getShift();
    DenseElementsAttr shiftElem;
    bool shiftIsConstant = true;
    int32_t shift = 0;
    if (matchPattern(shiftVal, m_Constant(&shiftElem)))
      shift = shiftElem.getValues<IntegerAttr>()[0].getInt();
    else
      shiftIsConstant = false;

    if (isa<FloatType>(elementTy)) {
      if (shift != 0) {
        (void)rewriter.notifyMatchFailure(op,
                                          "Cannot have shift value for float");
        return nullptr;
      }
      return arith::MulFOp::create(rewriter, loc, resultTypes, args[0],
                                   args[1]);
    }

    if (isa<IntegerType>(elementTy)) {
      Value a = args[0];
      Value b = args[1];

      if (shift > 0 || !shiftIsConstant) {
        Value shiftConst;
        if (shiftIsConstant)
          shiftConst = arith::ConstantIntOp::create(rewriter, loc, shift,
                                                    /*bitwidth=*/8);

        if (!a.getType().isInteger(32))
          a = arith::ExtSIOp::create(rewriter, loc, rewriter.getI32Type(), a);

        if (!b.getType().isInteger(32))
          b = arith::ExtSIOp::create(rewriter, loc, rewriter.getI32Type(), b);

        auto shiftAmount = shiftIsConstant ? shiftConst : args[2];
        auto roundingAttr = RoundingModeAttr::get(rewriter.getContext(),
                                                  RoundingMode::SINGLE_ROUND);
        auto result =
            tosa::ApplyScaleOp::create(rewriter, loc, rewriter.getI32Type(), a,
                                       b, shiftAmount, roundingAttr);

        return result;
      }

      int aWidth = a.getType().getIntOrFloatBitWidth();
      int bWidth = b.getType().getIntOrFloatBitWidth();
      int cWidth = resultTypes[0].getIntOrFloatBitWidth();

      if (aWidth < cWidth)
        a = arith::ExtSIOp::create(rewriter, loc, resultTypes[0], a);
      if (bWidth < cWidth)
        b = arith::ExtSIOp::create(rewriter, loc, resultTypes[0], b);

      return arith::MulIOp::create(rewriter, loc, resultTypes, a, b);
    }
  }

  // tosa::NegateOp
  if (isa<tosa::NegateOp>(op)) {
    auto negate = cast<tosa::NegateOp>(op);

    int64_t inZp = 0, outZp = 0;
    FailureOr<int64_t> maybeInZp = negate.getInput1ZeroPoint();
    FailureOr<int64_t> maybeOutZp = negate.getOutputZeroPoint();
    bool hasInZp = !failed(maybeInZp);
    bool hasOutZp = !failed(maybeOutZp);
    if (hasInZp)
      inZp = *maybeInZp;
    if (hasOutZp)
      outZp = *maybeOutZp;

    if (isa<FloatType>(elementTy))
      return arith::NegFOp::create(rewriter, loc, resultTypes, args[0]);

    if (isa<IntegerType>(elementTy)) {
      if (hasInZp && hasOutZp && !inZp && !outZp) {
        auto constant = arith::ConstantOp::create(
            rewriter, loc, IntegerAttr::get(elementTy, 0));
        return arith::SubIOp::create(rewriter, loc, resultTypes, constant,
                                     args[0]);
      }

      Value zpAddValue;
      Type intermediateType;
      // Compute the maximum value that can occur in the intermediate buffer.
      const int32_t inputBitWidth = elementTy.getIntOrFloatBitWidth();
      int intermediateBitWidth = 64;

      if (hasInZp && hasOutZp) {
        // Compute the maximum value that can occur in the intermediate buffer.
        const int64_t zpAdd = inZp + outZp;
        const int64_t maxValue =
            APInt::getSignedMaxValue(inputBitWidth).getSExtValue() +
            std::abs(zpAdd) + 1;

        // Convert that maximum value into the maximum bitwidth needed to
        // represent it. We assume 48-bit numbers may be supported further in
        // the pipeline.
        if (maxValue <= APInt::getSignedMaxValue(16).getSExtValue()) {
          intermediateBitWidth = 16;
        } else if (maxValue <= APInt::getSignedMaxValue(32).getSExtValue()) {
          intermediateBitWidth = 32;
        } else if (maxValue <= APInt::getSignedMaxValue(48).getSExtValue()) {
          intermediateBitWidth = 48;
        }

        intermediateType = rewriter.getIntegerType(intermediateBitWidth);
        zpAddValue = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(intermediateType, zpAdd));
      } else {
        intermediateType = rewriter.getIntegerType(intermediateBitWidth);
        auto arg1 =
            rewriter.create<arith::ExtSIOp>(loc, intermediateType, args[1]);
        auto arg2 =
            rewriter.create<arith::ExtSIOp>(loc, intermediateType, args[2]);
        zpAddValue =
            rewriter.create<arith::AddIOp>(loc, intermediateType, arg1, arg2);
      }

      // The negation can be applied by doing:
      //  outputValue = inZp + outZp - inputValue
      auto ext =
          arith::ExtSIOp::create(rewriter, loc, intermediateType, args[0]);
      auto sub = arith::SubIOp::create(rewriter, loc, zpAddValue, ext);

      // Clamp to the negation range.
      Value min = arith::ConstantIntOp::create(
          rewriter, loc, intermediateType,
          APInt::getSignedMinValue(inputBitWidth).getSExtValue());
      Value max = arith::ConstantIntOp::create(
          rewriter, loc, intermediateType,
          APInt::getSignedMaxValue(inputBitWidth).getSExtValue());
      auto clamp = clampIntHelper(loc, sub, min, max, rewriter, false);

      // Truncate to the final value.
      return arith::TruncIOp::create(rewriter, loc, elementTy, clamp);
    }
  }

  // tosa::BitwiseAndOp
  if (isa<tosa::BitwiseAndOp>(op) && isa<IntegerType>(elementTy))
    return arith::AndIOp::create(rewriter, loc, resultTypes, args);

  // tosa::BitwiseOrOp
  if (isa<tosa::BitwiseOrOp>(op) && isa<IntegerType>(elementTy))
    return arith::OrIOp::create(rewriter, loc, resultTypes, args);

  // tosa::BitwiseNotOp
  if (isa<tosa::BitwiseNotOp>(op) && isa<IntegerType>(elementTy)) {
    auto allOnesAttr = rewriter.getIntegerAttr(
        elementTy, APInt::getAllOnes(elementTy.getIntOrFloatBitWidth()));
    auto allOnes = arith::ConstantOp::create(rewriter, loc, allOnesAttr);
    return arith::XOrIOp::create(rewriter, loc, resultTypes, args[0], allOnes);
  }

  // tosa::BitwiseXOrOp
  if (isa<tosa::BitwiseXorOp>(op) && isa<IntegerType>(elementTy))
    return arith::XOrIOp::create(rewriter, loc, resultTypes, args);

  // tosa::LogicalLeftShiftOp
  if (isa<tosa::LogicalLeftShiftOp>(op) && isa<IntegerType>(elementTy))
    return arith::ShLIOp::create(rewriter, loc, resultTypes, args);

  // tosa::LogicalRightShiftOp
  if (isa<tosa::LogicalRightShiftOp>(op) && isa<IntegerType>(elementTy))
    return arith::ShRUIOp::create(rewriter, loc, resultTypes, args);

  // tosa::ArithmeticRightShiftOp
  if (isa<tosa::ArithmeticRightShiftOp>(op) && isa<IntegerType>(elementTy)) {
    auto result = arith::ShRSIOp::create(rewriter, loc, resultTypes, args);
    auto round = cast<BoolAttr>(op->getAttr("round")).getValue();
    if (!round) {
      return result;
    }

    Type i1Ty = IntegerType::get(rewriter.getContext(), /*width=*/1);
    auto one = arith::ConstantOp::create(rewriter, loc,
                                         IntegerAttr::get(elementTy, 1));
    auto zero = arith::ConstantOp::create(rewriter, loc,
                                          IntegerAttr::get(elementTy, 0));
    auto i1zero =
        arith::ConstantOp::create(rewriter, loc, IntegerAttr::get(i1Ty, 0));
    auto i1one =
        arith::ConstantOp::create(rewriter, loc, IntegerAttr::get(i1Ty, 1));

    // Checking that input2 != 0
    auto shiftValueGreaterThanZero = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sgt, args[1], zero);

    // Checking for the last bit of input1 to be 1
    auto subtract =
        arith::SubIOp::create(rewriter, loc, resultTypes, args[1], one);
    auto shifted =
        arith::ShRSIOp::create(rewriter, loc, resultTypes, args[0], subtract)
            ->getResults();
    auto truncated = arith::TruncIOp::create(rewriter, loc, i1Ty, shifted,
                                             ArrayRef<NamedAttribute>());
    auto isInputOdd =
        arith::AndIOp::create(rewriter, loc, i1Ty, truncated, i1one);
    // shifted, truncated, isInputOdd can be poison when input2 is 0.
    auto shouldRound = arith::SelectOp::create(
        rewriter, loc, i1Ty, shiftValueGreaterThanZero, isInputOdd, i1zero);
    auto extended =
        arith::ExtUIOp::create(rewriter, loc, resultTypes, shouldRound);
    return arith::AddIOp::create(rewriter, loc, resultTypes, result, extended);
  }

  // tosa::ClzOp
  if (isa<tosa::ClzOp>(op) && isa<IntegerType>(elementTy)) {
    return math::CountLeadingZerosOp::create(rewriter, loc, elementTy, args[0]);
  }

  // tosa::LogicalAnd
  if (isa<tosa::LogicalAndOp>(op) && elementTy.isInteger(1))
    return arith::AndIOp::create(rewriter, loc, resultTypes, args);

  // tosa::LogicalNot
  if (isa<tosa::LogicalNotOp>(op) && elementTy.isInteger(1)) {
    auto one = arith::ConstantOp::create(rewriter, loc,
                                         rewriter.getIntegerAttr(elementTy, 1));
    return arith::XOrIOp::create(rewriter, loc, resultTypes, args[0], one);
  }

  // tosa::LogicalOr
  if (isa<tosa::LogicalOrOp>(op) && elementTy.isInteger(1))
    return arith::OrIOp::create(rewriter, loc, resultTypes, args);

  // tosa::LogicalXor
  if (isa<tosa::LogicalXorOp>(op) && elementTy.isInteger(1))
    return arith::XOrIOp::create(rewriter, loc, resultTypes, args);

  // tosa::PowOp
  if (isa<tosa::PowOp>(op) && isa<FloatType>(elementTy))
    return mlir::math::PowFOp::create(rewriter, loc, resultTypes, args);

  // tosa::RsqrtOp
  if (isa<tosa::RsqrtOp>(op) && isa<FloatType>(elementTy))
    return mlir::math::RsqrtOp::create(rewriter, loc, resultTypes, args);

  // tosa::LogOp
  if (isa<tosa::LogOp>(op) && isa<FloatType>(elementTy))
    return mlir::math::LogOp::create(rewriter, loc, resultTypes, args);

  // tosa::ExpOp
  if (isa<tosa::ExpOp>(op) && isa<FloatType>(elementTy))
    return mlir::math::ExpOp::create(rewriter, loc, resultTypes, args);

  // tosa::SinOp
  if (isa<tosa::SinOp>(op) && isa<FloatType>(elementTy))
    return mlir::math::SinOp::create(rewriter, loc, resultTypes, args);

  // tosa::CosOp
  if (isa<tosa::CosOp>(op) && isa<FloatType>(elementTy))
    return mlir::math::CosOp::create(rewriter, loc, resultTypes, args);

  // tosa::TanhOp
  if (isa<tosa::TanhOp>(op) && isa<FloatType>(elementTy))
    return mlir::math::TanhOp::create(rewriter, loc, resultTypes, args);

  // tosa::ErfOp
  if (isa<tosa::ErfOp>(op) && llvm::isa<FloatType>(elementTy))
    return mlir::math::ErfOp::create(rewriter, loc, resultTypes, args);

  // tosa::GreaterOp
  if (isa<tosa::GreaterOp>(op) && isa<FloatType>(elementTy))
    return arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGT,
                                 args[0], args[1]);

  if (isa<tosa::GreaterOp>(op) && elementTy.isSignlessInteger())
    return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                                 args[0], args[1]);

  // tosa::GreaterEqualOp
  if (isa<tosa::GreaterEqualOp>(op) && isa<FloatType>(elementTy))
    return arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGE,
                                 args[0], args[1]);

  if (isa<tosa::GreaterEqualOp>(op) && elementTy.isSignlessInteger())
    return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge,
                                 args[0], args[1]);

  // tosa::EqualOp
  if (isa<tosa::EqualOp>(op) && isa<FloatType>(elementTy))
    return arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OEQ,
                                 args[0], args[1]);

  if (isa<tosa::EqualOp>(op) && elementTy.isSignlessInteger())
    return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                 args[0], args[1]);

  // tosa::SelectOp
  if (isa<tosa::SelectOp>(op)) {
    elementTy = cast<ShapedType>(op->getOperand(1).getType()).getElementType();
    if (isa<FloatType>(elementTy) || isa<IntegerType>(elementTy))
      return arith::SelectOp::create(rewriter, loc, args[0], args[1], args[2]);
  }

  // tosa::MaximumOp
  if (isa<tosa::MaximumOp>(op) && isa<FloatType>(elementTy)) {
    auto max = arith::MaximumFOp::create(rewriter, loc, args[0], args[1]);
    return materializeBinaryNanCheckIfRequired(llvm::cast<tosa::MaximumOp>(op),
                                               rewriter, args[0], args[1], max);
  }

  if (isa<tosa::MaximumOp>(op) && elementTy.isSignlessInteger()) {
    return arith::MaxSIOp::create(rewriter, loc, args[0], args[1]);
  }

  // tosa::MinimumOp
  if (isa<tosa::MinimumOp>(op) && isa<FloatType>(elementTy)) {
    auto min = arith::MinimumFOp::create(rewriter, loc, args[0], args[1]);
    return materializeBinaryNanCheckIfRequired(llvm::cast<tosa::MinimumOp>(op),
                                               rewriter, args[0], args[1], min);
  }

  if (isa<tosa::MinimumOp>(op) && elementTy.isSignlessInteger()) {
    return arith::MinSIOp::create(rewriter, loc, args[0], args[1]);
  }

  // tosa::CeilOp
  if (isa<tosa::CeilOp>(op) && isa<FloatType>(elementTy))
    return math::CeilOp::create(rewriter, loc, resultTypes, args);

  // tosa::FloorOp
  if (isa<tosa::FloorOp>(op) && isa<FloatType>(elementTy))
    return math::FloorOp::create(rewriter, loc, resultTypes, args);

  // tosa::ClampOp
  if (isa<tosa::ClampOp>(op) && isa<FloatType>(elementTy)) {
    bool losesInfo = false;
    APFloat minApf = cast<FloatAttr>(op->getAttr("min_val")).getValue();
    APFloat maxApf = cast<FloatAttr>(op->getAttr("max_val")).getValue();
    minApf.convert(cast<FloatType>(elementTy).getFloatSemantics(),
                   APFloat::rmNearestTiesToEven, &losesInfo);
    maxApf.convert(cast<FloatType>(elementTy).getFloatSemantics(),
                   APFloat::rmNearestTiesToEven, &losesInfo);
    auto min = arith::ConstantOp::create(
        rewriter, loc, elementTy, rewriter.getFloatAttr(elementTy, minApf));
    auto max = arith::ConstantOp::create(
        rewriter, loc, elementTy, rewriter.getFloatAttr(elementTy, maxApf));
    auto result = clampFloatHelper(loc, args[0], min, max, rewriter);

    auto clampOp = llvm::cast<tosa::ClampOp>(op);
    const auto nanMode = clampOp.getNanMode();

    // NaN propagation has no meaning for non floating point types.
    if (!isa<FloatType>(elementTy))
      return result;

    // In the case of "PROPAGATE" semantics no compare and selection is
    // required.
    if (nanMode == NanPropagationMode::PROPAGATE)
      return result;

    // In the case of "IGNORE" semantics materialize a comparison
    // of the current operand to the reduction which will return true for a NaN
    // argument and then selects between the initial reduction value and the
    // calculated result based on whether the argument is NaN or not. In pseudo
    // code:
    //
    // reduce<op>(x, init):
    //   result = op(init, x)
    //   return init if x == NaN else result

    // Unordered comparison of NaN against itself will always return true.
    Value isNaN = arith::CmpFOp::create(
        rewriter, op->getLoc(), arith::CmpFPredicate::UNO, args[0], args[0]);
    // TOSA specifies that in "ignore" NaN mode the result is "min" if the input
    // is NaN.
    return arith::SelectOp::create(rewriter, op->getLoc(), isNaN, min, result);
  }

  if (isa<tosa::ClampOp>(op) && isa<IntegerType>(elementTy)) {
    auto intTy = cast<IntegerType>(elementTy);
    int64_t min =
        cast<IntegerAttr>(op->getAttr("min_val")).getValue().getSExtValue();
    int64_t max =
        cast<IntegerAttr>(op->getAttr("max_val")).getValue().getSExtValue();

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

    auto minVal = arith::ConstantIntOp::create(rewriter, loc, min,
                                               intTy.getIntOrFloatBitWidth());
    auto maxVal = arith::ConstantIntOp::create(rewriter, loc, max,
                                               intTy.getIntOrFloatBitWidth());
    return clampIntHelper(loc, args[0], minVal, maxVal, rewriter,
                          intTy.isUnsignedInteger());
  }

  // tosa::SigmoidOp
  if (isa<tosa::SigmoidOp>(op) && isa<FloatType>(elementTy)) {
    auto one =
        arith::ConstantOp::create(rewriter, loc, FloatAttr::get(elementTy, 1));
    auto negate = arith::NegFOp::create(rewriter, loc, resultTypes, args[0]);
    auto exp = mlir::math::ExpOp::create(rewriter, loc, resultTypes, negate);
    auto added = arith::AddFOp::create(rewriter, loc, resultTypes, exp, one);
    return arith::DivFOp::create(rewriter, loc, resultTypes, one, added);
  }

  // tosa::CastOp
  if (isa<tosa::CastOp>(op)) {
    Type srcTy = elementTy;
    Type dstTy = resultTypes.front();
    if (!srcTy.isIntOrFloat() || !dstTy.isIntOrFloat()) {
      (void)rewriter.notifyMatchFailure(op, "unsupported type");
      return nullptr;
    }

    bool bitExtend =
        srcTy.getIntOrFloatBitWidth() < dstTy.getIntOrFloatBitWidth();

    if (srcTy == dstTy)
      return args.front();

    if (isa<FloatType>(srcTy) && isa<FloatType>(dstTy) && bitExtend)
      return arith::ExtFOp::create(rewriter, loc, resultTypes, args,
                                   ArrayRef<NamedAttribute>());

    if (isa<FloatType>(srcTy) && isa<FloatType>(dstTy) && !bitExtend)
      return arith::TruncFOp::create(rewriter, loc, resultTypes, args,
                                     ArrayRef<NamedAttribute>());

    // 1-bit integers need to be treated as signless.
    if (srcTy.isInteger(1) && arith::UIToFPOp::areCastCompatible(srcTy, dstTy))
      return arith::UIToFPOp::create(rewriter, loc, resultTypes, args,
                                     ArrayRef<NamedAttribute>());

    if (srcTy.isInteger(1) && isa<IntegerType>(dstTy) && bitExtend)
      return arith::ExtUIOp::create(rewriter, loc, resultTypes, args,
                                    ArrayRef<NamedAttribute>());

    // Unsigned integers need an unrealized cast so that they can be passed
    // to UIToFP.
    if (srcTy.isUnsignedInteger() && isa<FloatType>(dstTy)) {
      auto unrealizedCast =
          UnrealizedConversionCastOp::create(
              rewriter, loc,
              rewriter.getIntegerType(srcTy.getIntOrFloatBitWidth()), args[0])
              .getResult(0);
      return arith::UIToFPOp::create(rewriter, loc, resultTypes[0],
                                     unrealizedCast);
    }

    // All other si-to-fp conversions should be handled by SIToFP.
    if (arith::SIToFPOp::areCastCompatible(srcTy, dstTy))
      return arith::SIToFPOp::create(rewriter, loc, resultTypes, args,
                                     ArrayRef<NamedAttribute>());

    // Casting to boolean, floats need to only be checked as not-equal to zero.
    if (isa<FloatType>(srcTy) && dstTy.isInteger(1)) {
      Value zero = arith::ConstantOp::create(rewriter, loc,
                                             rewriter.getFloatAttr(srcTy, 0.0));
      return arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::UNE,
                                   args.front(), zero);
    }

    if (arith::FPToSIOp::areCastCompatible(srcTy, dstTy)) {
      auto rounded = math::RoundEvenOp::create(rewriter, loc, args[0]);

      const auto &fltSemantics = cast<FloatType>(srcTy).getFloatSemantics();
      // Check whether neither int min nor int max can be represented in the
      // input floating-point type due to too short exponent range.
      if (static_cast<int>(dstTy.getIntOrFloatBitWidth()) - 1 >
          APFloat::semanticsMaxExponent(fltSemantics)) {
        // Use cmp + select to replace infinites by int min / int max. Other
        // integral values can be represented in the integer space.
        auto conv = arith::FPToSIOp::create(rewriter, loc, dstTy, rounded);
        auto posInf = arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getFloatAttr(getElementTypeOrSelf(srcTy),
                                  APFloat::getInf(fltSemantics)));
        auto negInf = arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getFloatAttr(
                getElementTypeOrSelf(srcTy),
                APFloat::getInf(fltSemantics, /*Negative=*/true)));
        auto overflow = arith::CmpFOp::create(
            rewriter, loc, arith::CmpFPredicate::UEQ, rounded, posInf);
        auto underflow = arith::CmpFOp::create(
            rewriter, loc, arith::CmpFPredicate::UEQ, rounded, negInf);
        auto intMin = arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getIntegerAttr(
                getElementTypeOrSelf(dstTy),
                APInt::getSignedMinValue(dstTy.getIntOrFloatBitWidth())));
        auto intMax = arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getIntegerAttr(
                getElementTypeOrSelf(dstTy),
                APInt::getSignedMaxValue(dstTy.getIntOrFloatBitWidth())));
        auto maxClamped =
            arith::SelectOp::create(rewriter, loc, overflow, intMax, conv);
        return arith::SelectOp::create(rewriter, loc, underflow, intMin,
                                       maxClamped);
      }

      auto intMinFP = arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getFloatAttr(
              getElementTypeOrSelf(srcTy),
              APInt::getSignedMinValue(dstTy.getIntOrFloatBitWidth())
                  .getSExtValue()));

      // Check whether the mantissa has enough bits to represent int max.
      if (cast<FloatType>(srcTy).getFPMantissaWidth() >=
          dstTy.getIntOrFloatBitWidth() - 1) {
        // Int min can also be represented since it is a power of two and thus
        // consists of a single leading bit. Therefore we can clamp the input
        // in the floating-point domain.

        auto intMaxFP = arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getFloatAttr(
                getElementTypeOrSelf(srcTy),
                APInt::getSignedMaxValue(dstTy.getIntOrFloatBitWidth())
                    .getSExtValue()));

        Value clamped =
            clampFloatHelper(loc, rounded, intMinFP, intMaxFP, rewriter);
        return arith::FPToSIOp::create(rewriter, loc, dstTy, clamped);
      }

      // Due to earlier check we know exponant range is big enough to represent
      // int min. We can therefore rely on int max + 1 being representable as
      // well because it's just int min with a positive sign. So clamp the min
      // value and compare against that to select the max int value if needed.
      auto intMaxPlusOneFP = arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getFloatAttr(
              getElementTypeOrSelf(srcTy),
              static_cast<double>(
                  APInt::getSignedMaxValue(dstTy.getIntOrFloatBitWidth())
                      .getSExtValue()) +
                  1.0f));

      auto intMax = arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getIntegerAttr(
              getElementTypeOrSelf(dstTy),
              APInt::getSignedMaxValue(dstTy.getIntOrFloatBitWidth())));
      auto minClampedFP =
          arith::MaximumFOp::create(rewriter, loc, rounded, intMinFP);
      auto minClamped =
          arith::FPToSIOp::create(rewriter, loc, dstTy, minClampedFP);
      auto overflow = arith::CmpFOp::create(
          rewriter, loc, arith::CmpFPredicate::UGE, rounded, intMaxPlusOneFP);
      return arith::SelectOp::create(rewriter, loc, overflow, intMax,
                                     minClamped);
    }

    // Casting to boolean, integers need to only be checked as not-equal to
    // zero.
    if (isa<IntegerType>(srcTy) && dstTy.isInteger(1)) {
      Value zero = arith::ConstantIntOp::create(rewriter, loc, 0,
                                                srcTy.getIntOrFloatBitWidth());
      return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                   args.front(), zero);
    }

    if (isa<IntegerType>(srcTy) && isa<IntegerType>(dstTy) && bitExtend)
      return arith::ExtSIOp::create(rewriter, loc, resultTypes, args,
                                    ArrayRef<NamedAttribute>());

    if (isa<IntegerType>(srcTy) && isa<IntegerType>(dstTy) && !bitExtend) {
      return arith::TruncIOp::create(rewriter, loc, dstTy, args[0]);
    }
  }

  (void)rewriter.notifyMatchFailure(
      op, "unhandled op for linalg body calculation for elementwise op");
  return nullptr;
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
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(index));
  return it->second;
}

static Value getTensorDim(PatternRewriter &rewriter, Location loc,
                          IndexPool &indexPool, Value tensor, int64_t index) {
  auto indexValue = createIndex(rewriter, loc, indexPool, index);
  return tensor::DimOp::create(rewriter, loc, tensor, indexValue).getResult();
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
    if (ShapedType::isStatic(size) && size > 1)
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
    targetSize = arith::MaxUIOp::create(rewriter, loc, targetSize, nextSize);
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
  auto broadcastNecessary = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::eq, runtimeSize, one);

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
    Value outputTensor = tensor::EmptyOp::create(
        opBuilder, loc, outputTensorShape, rankedTensorType.getElementType());

    // Emit 'linalg.generic' op
    auto resultTensor =
        linalg::GenericOp::create(
            opBuilder, loc, outputTensor.getType(), operand, outputTensor,
            affineMaps, getNParallelLoopsAttrs(rank),
            [&](OpBuilder &opBuilder, Location loc, ValueRange blockArgs) {
              // Emit 'linalg.yield' op
              linalg::YieldOp::create(opBuilder, loc, blockArgs.front());
            })
            .getResult(0);

    // Cast to original operand type if necessary
    auto castResultTensor = rewriter.createOrFold<tensor::CastOp>(
        loc, operand.getType(), resultTensor);

    // Emit 'scf.yield' op
    scf::YieldOp::create(opBuilder, loc, castResultTensor);
  };

  // Emit 'else' region of 'scf.if'
  auto emitElseRegion = [&](OpBuilder &opBuilder, Location loc) {
    scf::YieldOp::create(opBuilder, loc, operand);
  };

  // Emit 'scf.if' op
  auto ifOp = scf::IfOp::create(rewriter, loc, broadcastNecessary,
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

  // No need to broadcast for static shape
  bool hasDynamic = false;
  for (auto op : operands) {
    const auto tType = dyn_cast<RankedTensorType>(op.getType());
    if (tType && !tType.hasStaticShape()) {
      hasDynamic = true;
      break;
    }
  }
  if (!hasDynamic)
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
  Value outputTensor = tensor::EmptyOp::create(rewriter, loc, targetShape,
                                               resultType.getElementType());

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
  auto linalgOp = linalg::GenericOp::create(
      rewriter, loc, outputTensor.getType(), operands, outputTensor, affineMaps,
      getNParallelLoopsAttrs(rank),
      [&](OpBuilder &opBuilder, Location loc, ValueRange blockArgs) {
        Value opResult = createLinalgBodyCalculationForElementwiseOp(
            operation, blockArgs.take_front(operation->getNumOperands()),
            {resultType.getElementType()}, rewriter);
        if (!opResult) {
          encounteredError = true;
          return;
        }
        linalg::YieldOp::create(opBuilder, loc, opResult);
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

static ValueRange getBroadcastableOperands(Operation *operation,
                                           ValueRange operands) {
  // Shift cannot broadcast
  if (isa<tosa::MulOp>(operation)) {
    DenseElementsAttr shiftElems;
    // Shift cannot broadcast when it is constant
    if (matchPattern(operation->getOperand(2), m_Constant(&shiftElems)))
      return operands.take_front(2);
    else
      return operands.take_front(3);
  }
  if (auto negate = dyn_cast<tosa::NegateOp>(operation)) {
    FailureOr<int64_t> maybeInZp = negate.getInput1ZeroPoint();
    FailureOr<int64_t> maybeOutZp = negate.getOutputZeroPoint();
    if (failed(maybeOutZp) && failed(maybeInZp))
      return operands;
    // Input1_zp and output_zp cannot broadcast when they are constants.
    return operands.take_front(1);
  }
  return operands;
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
  auto operandsToBroadcast = getBroadcastableOperands(operation, operands);
  auto [targetShape, masterOperands] =
      computeTargetShape(rewriter, loc, indexPool, operandsToBroadcast);
  auto broadcastOperands =
      broadcastDynamicDimensions(rewriter, loc, indexPool, operandsToBroadcast,
                                 targetShape, masterOperands);
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

  if (isa<tosa::ReduceProductOp>(op) && isa<FloatType>(elementTy))
    return rewriter.getFloatAttr(elementTy, 1.0);

  if (isa<tosa::ReduceProductOp>(op) && isa<IntegerType>(elementTy))
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
    return arith::AddFOp::create(rewriter, loc, args);
  }

  if (isa<tosa::ReduceSumOp>(op) && isa<IntegerType>(elementTy)) {
    return arith::AddIOp::create(rewriter, loc, args);
  }

  if (isa<tosa::ReduceProductOp>(op) && isa<FloatType>(elementTy)) {
    return arith::MulFOp::create(rewriter, loc, args);
  }

  if (isa<tosa::ReduceProductOp>(op) && isa<IntegerType>(elementTy)) {
    return arith::MulIOp::create(rewriter, loc, args);
  }

  if (isa<tosa::ReduceMinOp>(op) && isa<FloatType>(elementTy)) {
    return arith::MinimumFOp::create(rewriter, loc, args[0], args[1]);
  }

  if (isa<tosa::ReduceMinOp>(op) && isa<IntegerType>(elementTy)) {
    return arith::MinSIOp::create(rewriter, loc, args[0], args[1]);
  }

  if (isa<tosa::ReduceMaxOp>(op) && isa<FloatType>(elementTy)) {
    return arith::MaximumFOp::create(rewriter, loc, args[0], args[1]);
  }

  if (isa<tosa::ReduceMaxOp>(op) && isa<IntegerType>(elementTy)) {
    return arith::MaxSIOp::create(rewriter, loc, args[0], args[1]);
  }

  if (isa<tosa::ReduceAllOp>(op) && elementTy.isInteger(1))
    return arith::AndIOp::create(rewriter, loc, args);

  if (isa<tosa::ReduceAnyOp>(op) && elementTy.isInteger(1))
    return arith::OrIOp::create(rewriter, loc, args);

  return {};
}

// Performs the match and rewrite for reduction operations. This includes
// declaring a correctly sized initial value, and the linalg.generic operation
// that reduces across the specified axis.
template <typename OpTy>
static LogicalResult reduceMatchAndRewriteHelper(OpTy op, uint64_t axis,
                                                 PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto inputTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!inputTy || !resultTy)
    return rewriter.notifyMatchFailure(op, "unranked tensors not supported");

  auto elementTy = resultTy.getElementType();
  Value input = op->getOperand(0);

  // Figure out the accType if needed
  bool widenAccTy = std::is_same_v<OpTy, tosa::ReduceSumOp> &&
                    isa<FloatType>(elementTy) &&
                    cast<FloatType>(elementTy).isBF16();
  Type accTy = widenAccTy ? rewriter.getF32Type() : elementTy;

  SmallVector<int64_t> reduceShape;
  SmallVector<Value> dynDims;
  for (unsigned i = 0; i < inputTy.getRank(); i++) {
    if (axis != i) {
      reduceShape.push_back(inputTy.getDimSize(i));
      if (inputTy.isDynamicDim(i))
        dynDims.push_back(tensor::DimOp::create(rewriter, loc, input, i));
    }
  }

  SmallVector<Value> inputs, outputs;
  inputs.push_back(input);

  // First fill the output buffer with the init value.
  auto emptyTensor =
      tensor::EmptyOp::create(rewriter, loc, reduceShape, accTy, dynDims)
          .getResult();

  auto fillValueAttr = createInitialValueForReduceOp(op, accTy, rewriter);
  if (!fillValueAttr)
    return rewriter.notifyMatchFailure(
        op, "No initial value found for reduction operation");

  auto fillValue = arith::ConstantOp::create(rewriter, loc, fillValueAttr);
  auto filledTensor =
      linalg::FillOp::create(rewriter, loc, ValueRange{fillValue},
                             ValueRange{emptyTensor})
          .result();
  outputs.push_back(filledTensor);

  bool isNanIgnoreMode = false;
  if constexpr (std::is_same_v<OpTy, tosa::ReduceMinOp> ||
                std::is_same_v<OpTy, tosa::ReduceMaxOp>) {
    // NaN propagation has no meaning for non floating point types.
    if (isa<FloatType>(elementTy) &&
        op.getNanMode() == NanPropagationMode::IGNORE) {
      isNanIgnoreMode = true;
      // Because the TOSA spec requires the result be NaN iff all elements in
      // the reduction are NaN we can't simply perform a compare and select.
      // Additionally we have to keep track of whether we've seen any non-NaN
      // values and then do a final select based on this predicate.
      auto trueAttr = rewriter.getBoolAttr(true);
      auto trueValue = arith::ConstantOp::create(rewriter, loc, trueAttr);
      auto emptyBoolTensor =
          tensor::EmptyOp::create(rewriter, loc, reduceShape,
                                  trueValue.getType(), dynDims)
              .getResult();
      auto allResultsNaNTensor =
          linalg::FillOp::create(rewriter, loc, ValueRange{trueValue},
                                 ValueRange{emptyBoolTensor})
              .result();
      // Note that because the linalg::ReduceOp has two variadic arguments
      // (inputs and outputs) and it has the SameVariadicOperandSize trait we
      // need to have the same number of inputs and outputs.
      //
      // The second input isn't actually used anywhere since the value used to
      // update the NaN flag is calculated inside the body of the reduction and
      // then used to update an out value.
      // In order to satisfy type constraints we just pass another copy of the
      // input here.
      inputs.push_back(input);
      outputs.push_back(allResultsNaNTensor);
    }
  }

  bool didEncounterError = false;
  linalg::LinalgOp linalgOp = linalg::ReduceOp::create(
      rewriter, loc, inputs, outputs, axis,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        std::array<Value, 2> binaryArgs{
            blockArgs[0], isNanIgnoreMode ? blockArgs[2] : blockArgs[1]};

        // If reduction type differs then extend (applicable to reduce_sum)
        if (binaryArgs[0].getType() != accTy)
          binaryArgs[0] = arith::ExtFOp::create(nestedBuilder, nestedLoc, accTy,
                                                binaryArgs[0]);

        auto result = createLinalgBodyCalculationForReduceOp(op, binaryArgs,
                                                             accTy, rewriter);
        if (result)
          didEncounterError = true;

        SmallVector<Value> resultsToYield;
        if (isNanIgnoreMode) {
          auto inputValue = blockArgs[0];
          auto initialValue = blockArgs[2];
          auto oldAllResultsNanFlagValue = blockArgs[3];

          // Unordered comparison of NaN against itself will always return true.
          Value isNaN = arith::CmpFOp::create(nestedBuilder, op->getLoc(),
                                              arith::CmpFPredicate::UNO,
                                              inputValue, inputValue);
          // If we've encountered a NaN, take the non-NaN value.
          auto selectOp = arith::SelectOp::create(nestedBuilder, op->getLoc(),
                                                  isNaN, initialValue, result);
          // Update the flag which keeps track of whether we have seen a non-NaN
          // value.
          auto newAllResultsNanFlagValue = arith::AndIOp::create(
              nestedBuilder, op->getLoc(), oldAllResultsNanFlagValue, isNaN);
          resultsToYield.push_back(selectOp);
          resultsToYield.push_back(newAllResultsNanFlagValue);
        } else {
          resultsToYield.push_back(result);
        }
        linalg::YieldOp::create(nestedBuilder, loc, resultsToYield);
      });

  if (!didEncounterError)
    return rewriter.notifyMatchFailure(
        op, "unable to create linalg.generic body for reduce op");

  if (isNanIgnoreMode) {
    // Materialize a check to see whether we encountered any non-NaN values, if
    // we didn't we need to select a tensor of NaNs since the result will just
    // be the initial identity value propagated through all the compares and
    // selects inside the reduction.

    // Create a tensor full of NaNs.
    auto nanValueAttr = rewriter.getFloatAttr(
        accTy,
        APFloat::getNaN(cast<FloatType>(elementTy).getFloatSemantics(), false));
    auto nanValue = arith::ConstantOp::create(rewriter, loc, nanValueAttr);
    auto emptyNanTensor =
        tensor::EmptyOp::create(rewriter, loc, reduceShape, accTy, dynDims)
            .getResult();
    auto nanFilledTensor =
        linalg::FillOp::create(rewriter, loc, ValueRange{nanValue},
                               ValueRange{emptyNanTensor})
            .result();

    // Create an empty tensor, non need to fill this since it will be
    // overwritten by the select.
    auto finalEmptyTensor =
        tensor::EmptyOp::create(rewriter, loc, reduceShape, accTy, dynDims)
            .getResult();

    // Do a selection between the tensors akin to:
    // result = NaN if "all results NaN" else result.
    SmallVector<Value> ins, outs;
    ins.push_back(linalgOp->getOpResult(1));
    ins.push_back(nanFilledTensor);
    ins.push_back(linalgOp->getResult(0));
    outs.push_back(finalEmptyTensor);
    auto linalgSelect =
        linalg::SelectOp::create(rewriter, op->getLoc(), ins, outs);
    linalgOp = linalgSelect;
  }

  // Truncate back to resultTy if needed
  Value reducedRes = linalgOp->getResult(0);
  if (widenAccTy) {
    auto resEmptyOp =
        tensor::EmptyOp::create(rewriter, loc, reduceShape, elementTy, dynDims)
            .getResult();

    const unsigned reducedRank =
        cast<ShapedType>(reducedRes.getType()).getRank();
    auto identityMap = rewriter.getMultiDimIdentityMap(reducedRank);
    reducedRes =
        linalg::GenericOp::create(
            rewriter, loc, resEmptyOp.getType(), ValueRange{reducedRes},
            ValueRange{resEmptyOp},
            ArrayRef<AffineMap>{identityMap, identityMap},
            getNParallelLoopsAttrs(reducedRank),
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
              Value truncf = arith::TruncFOp::create(nestedBuilder, nestedLoc,
                                                     elementTy, args[0]);
              linalg::YieldOp::create(nestedBuilder, nestedLoc, truncf);
            })
            .getResults()[0];
  }

  SmallVector<ReassociationExprs, 4> reassociationMap;
  uint64_t expandInputRank = cast<ShapedType>(reducedRes.getType()).getRank();
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
  rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(op, resultTy, reducedRes,
                                                     reassociationMap);
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

// Collapse tensor<1xiN> into tensor<iN>
// E.g. tensor.collapse_shape %arg1 [] : tensor<1xi16> into tensor<i16>
static Value collapse1xNTensorToN(PatternRewriter &rewriter, Value input,
                                  Location loc) {
  SmallVector<ReassociationExprs, 1> reassociation;
  // Create the collapsed type
  auto inputType = cast<RankedTensorType>(input.getType());
  auto elemType = inputType.getElementType();
  auto collapsedType = RankedTensorType::get({}, elemType);
  // Emit the collapse op
  return rewriter.create<tensor::CollapseShapeOp>(loc, collapsedType, input,
                                                  reassociation);
}

static llvm::SmallVector<int8_t>
convertToI8(const llvm::SmallVector<int32_t> &input) {
  llvm::SmallVector<int8_t> output;
  output.reserve(input.size());

  for (auto v : llvm::map_range(
           input, [](int32_t val) { return static_cast<int8_t>(val); })) {
    output.push_back(v);
  }
  return output;
}

// The shift or multiplier may be either constant or non-constant, depending on
// whether dynamic extension is enabled.
// - If the shift or multiplier is non-constant, add it as an input to
// linalg::GenericOp by:
//     1. Pushing it into 'genericInputs'.
//     2. Appending a corresponding affine map to 'indexingMaps'.
// - If the shift or multiplier is constant, set 'constant' instead.
static void setupLinalgGenericOpInputAndIndexingMap(
    PatternRewriter &rewriter, llvm::SmallVector<int32_t> &values,
    SmallVector<Value, 4> &genericInputs, SmallVector<AffineMap> &indexingMaps,
    bool isConstant, tosa::RescaleOp op, Value &constant, int64_t &arg,
    bool isShift = false) {

  auto loc = op.getLoc();
  auto inputTy = cast<ShapedType>(op.getInput().getType());
  unsigned rank = inputTy.getRank();
  SmallVector<AffineExpr, 2> exprs = {rewriter.getAffineDimExpr(rank - 1)};

  if (isConstant) {
    // If we are rescaling per-channel then we need to store the
    // values in a buffer.
    if (values.size() == 1) {
      IntegerAttr intAttr = isShift
                                ? rewriter.getI8IntegerAttr(values.front())
                                : rewriter.getI32IntegerAttr(values.front());
      constant = rewriter.create<arith::ConstantOp>(loc, intAttr);
    } else {
      auto elementType =
          isShift ? rewriter.getIntegerType(8) : rewriter.getI32Type();
      auto tensorType = RankedTensorType::get(
          {static_cast<int64_t>(values.size())}, elementType);
      DenseIntElementsAttr EltAttr;
      if (isShift)
        EltAttr = DenseIntElementsAttr::get(tensorType, convertToI8(values));
      else
        EltAttr = DenseIntElementsAttr::get(tensorType, values);
      genericInputs.push_back(
          arith::ConstantOp::create(rewriter, loc, EltAttr));
      indexingMaps.push_back(AffineMap::get(/*dimCount=*/rank,
                                            /*symbolCount=*/0, exprs,
                                            rewriter.getContext()));
    }
  } else {
    // If we are not rescaling per-channel then we need to collapse 1xN to N
    // and push broadcastMap.
    auto operand = isShift ? op.getShift() : op.getMultiplier();
    auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
    if (tensorType && tensorType.hasStaticShape() &&
        tensorType.getShape()[0] == 1) {
      // broadcastMap = affine_map<(d0, d1) -> ()>
      // It would affect as broadcast for scalar values in linalg::GenericOp.
      AffineMap broadcastMap =
          AffineMap::get(rank, 0, {}, rewriter.getContext());
      genericInputs.push_back(collapse1xNTensorToN(rewriter, operand, loc));
      indexingMaps.push_back(broadcastMap);
    } else {
      genericInputs.push_back(operand);
      indexingMaps.push_back(AffineMap::get(/*dimCount=*/rank,
                                            /*symbolCount=*/0, exprs,
                                            rewriter.getContext()));
    }
  }
  arg = indexingMaps.size() - 1;
}

// Return the extended Zp to be used in subsequent arithmetic operations.
static Value getExtendZp(OpBuilder &builder, Type valueTy,
                         FailureOr<int64_t> maybeZp, Location loc,
                         ValueRange blockArgs, int64_t zpArg,
                         bool isOutputZp = false) {
  Value result;
  const int32_t bitwidth = valueTy.getIntOrFloatBitWidth();
  const uint32_t attrBitwidth =
      isOutputZp ? 32 : (bitwidth > 32 ? bitwidth : 32);
  auto extendType = builder.getIntegerType(attrBitwidth);
  // The Zp value can be either constant or non-constant, depending on
  // whether dynamic extension is enabled.
  // If 'maybeZp' fails, it indicates that Zp is non-constant and will
  // be passed as an input to linalg::GenericOp.
  if (failed(maybeZp)) {
    result = blockArgs[zpArg];
    auto zpTy = result.getType();
    if (zpTy.getIntOrFloatBitWidth() < attrBitwidth) {
      // For ExtUIOp, the input must be signless.
      // UnrealizedConversionCastOp will cast the input to signless type.
      if (zpTy.isUnsignedInteger()) {
        result =
            UnrealizedConversionCastOp::create(
                builder, loc,
                builder.getIntegerType(zpTy.getIntOrFloatBitWidth()), result)
                .getResult(0);
      }
      if (zpTy.isUnsignedInteger()) {
        return builder.create<arith::ExtUIOp>(loc, extendType, result);
      } else {
        return builder.create<arith::ExtSIOp>(loc, extendType, result);
      }
    }
  } else {
    return builder.create<arith::ConstantOp>(
        loc, IntegerAttr::get(extendType, *maybeZp));
  }
  return result;
}

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
    if (op.getRoundingMode() == RoundingMode::INEXACT_ROUND)
      return rewriter.notifyMatchFailure(
          op, "tosa.rescale with rounding mode = 'INEXACT_ROUND' is not "
              "currently supported");
    if (op.getRoundingMode() == RoundingMode::DOUBLE_ROUND && !op.getScale32())
      return rewriter.notifyMatchFailure(
          op, "tosa.rescale requires scale32 for double_round to be true");

    if (!isa<IntegerType>(inputTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "only support integer type");

    SmallVector<Value> dynDims;
    for (int i = 0; i < outputTy.getRank(); i++) {
      if (outputTy.isDynamicDim(i)) {
        dynDims.push_back(tensor::DimOp::create(rewriter, loc, input, i));
      }
    }

    DenseElementsAttr shiftElems;
    bool isShiftConstant = false;
    if (matchPattern(op.getShift(), m_Constant(&shiftElems)))
      isShiftConstant = true;

    DenseElementsAttr multiplierElems;
    bool isMultiplierConstant = false;
    if (matchPattern(op.getMultiplier(), m_Constant(&multiplierElems)))
      isMultiplierConstant = true;

    llvm::SmallVector<int32_t> shiftValues;
    llvm::SmallVector<int32_t> multiplierValues;
    bool doubleRound;

    if (isMultiplierConstant && isShiftConstant) {
      // explicit cast is required here
      shiftValues = llvm::to_vector(llvm::map_range(
          shiftElems.getValues<IntegerAttr>(), [](IntegerAttr attr) -> int32_t {
            return static_cast<int32_t>(attr.getInt());
          }));
      multiplierValues = llvm::to_vector(
          llvm::map_range(multiplierElems.getValues<IntegerAttr>(),
                          [](IntegerAttr attr) -> int32_t {
                            return static_cast<int32_t>(attr.getInt());
                          }));

      // If we shift by more than the bitwidth, this just sets to 0.
      for (int i = 0, s = multiplierValues.size(); i < s; i++) {
        if (shiftValues[i] > 63) {
          shiftValues[i] = 0;
          multiplierValues[i] = 0;
        }
      }
      // Double round only occurs if shift is greater than 31, check that this
      // is ever true.
      doubleRound = op.getRoundingMode() == RoundingMode::DOUBLE_ROUND &&
                    llvm::any_of(shiftValues, [](int32_t v) { return v > 31; });
    } else
      doubleRound = op.getRoundingMode() == RoundingMode::DOUBLE_ROUND;

    RoundingMode roundingMode =
        doubleRound ? RoundingMode::DOUBLE_ROUND : RoundingMode::SINGLE_ROUND;

    SmallVector<AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(rank)};
    SmallVector<Value, 4> genericInputs = {input};

    // If we are rescaling per-channel then we need to store the multiplier
    // values in a buffer.
    Value multiplierConstant;
    int64_t multiplierArg = 0;
    setupLinalgGenericOpInputAndIndexingMap(
        rewriter, multiplierValues, genericInputs, indexingMaps,
        isMultiplierConstant, op, multiplierConstant, multiplierArg);

    // If we are rescaling per-channel then we need to store the shift
    // values in a buffer.
    Value shiftConstant;
    int64_t shiftArg = 0;
    setupLinalgGenericOpInputAndIndexingMap(
        rewriter, shiftValues, genericInputs, indexingMaps, isShiftConstant, op,
        shiftConstant, shiftArg, true);

    // broadcastMap = affine_map<(d0, d1) -> ()>
    // It would affect as broadcast for scalar values in linalg::GenericOp.
    AffineMap broadcastMap = AffineMap::get(rank, 0, {}, rewriter.getContext());
    FailureOr<int64_t> maybeIZp = op.getInputZeroPoint();
    FailureOr<int64_t> maybeOZp = op.getOutputZeroPoint();
    // The inputZp and outputZp may be either constant or non-constant,
    // depending on whether dynamic extension is enabled.
    // - If the zp's are non-constant, add them as an inputs to
    // linalg::GenericOp by:
    //     1. Pushing it into 'genericInputs'.
    //     2. Appending a corresponding affine map to 'indexingMaps'.
    // - If the zp's are constant, they would be generated as arith.constant.
    int64_t iZpArg = 0;
    if (failed(maybeIZp)) {
      genericInputs.push_back(
          collapse1xNTensorToN(rewriter, op->getOperand(3), loc));
      indexingMaps.push_back(broadcastMap);
      iZpArg = indexingMaps.size() - 1;
    }
    int64_t oZpArg = 0;
    if (failed(maybeOZp)) {
      genericInputs.push_back(
          collapse1xNTensorToN(rewriter, op->getOperand(4), loc));
      indexingMaps.push_back(broadcastMap);
      oZpArg = indexingMaps.size() - 1;
    }

    // Indexing maps for output values.
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));

    // Construct the indexing maps needed for linalg.generic ops.
    Value emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, outputTy.getShape(), outputTy.getElementType(),
        ArrayRef<Value>({dynDims}));

    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, outputTy, genericInputs, ValueRange{emptyTensor},
        indexingMaps, getNParallelLoopsAttrs(rank),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value value = blockArgs[0];
          Type valueTy = value.getType();

          FailureOr<int64_t> maybeIZp = op.getInputZeroPoint();
          auto inputZp = getExtendZp(nestedBuilder, valueTy, maybeIZp,
                                     nestedLoc, blockArgs, iZpArg);

          FailureOr<int64_t> maybeOZp = op.getOutputZeroPoint();
          auto outputZp = getExtendZp(nestedBuilder, valueTy, maybeOZp,
                                      nestedLoc, blockArgs, oZpArg, true);

          IntegerType outIntType =
              cast<IntegerType>(blockArgs.back().getType());
          unsigned outBitWidth = outIntType.getWidth();
          assert(outBitWidth <= 32 && "Unexpected output zeropoint bitwidth");

          Value multiplier = multiplierConstant ? multiplierConstant
                                                : blockArgs[multiplierArg];
          Value shift = shiftConstant ? shiftConstant : blockArgs[shiftArg];

          if (valueTy.isUnsignedInteger()) {
            value = UnrealizedConversionCastOp::create(
                        nestedBuilder, nestedLoc,
                        nestedBuilder.getIntegerType(
                            valueTy.getIntOrFloatBitWidth()),
                        value)
                        .getResult(0);
          }
          if (valueTy.getIntOrFloatBitWidth() < 32) {
            if (op.getInputUnsigned()) {
              value = arith::ExtUIOp::create(nestedBuilder, nestedLoc,
                                             nestedBuilder.getI32Type(), value);
            } else {
              value = arith::ExtSIOp::create(nestedBuilder, nestedLoc,
                                             nestedBuilder.getI32Type(), value);
            }
          }

          value =
              arith::SubIOp::create(nestedBuilder, nestedLoc, value, inputZp);

          value = tosa::ApplyScaleOp::create(nestedBuilder, loc,
                                             nestedBuilder.getI32Type(), value,
                                             multiplier, shift, roundingMode);

          // Move to the new zero-point.
          value =
              arith::AddIOp::create(nestedBuilder, nestedLoc, value, outputZp);

          // Saturate to the output size.
          int32_t intMin = APInt::getSignedMinValue(outBitWidth).getSExtValue();
          int32_t intMax = APInt::getSignedMaxValue(outBitWidth).getSExtValue();

          // Unsigned integers have a difference output value.
          if (op.getOutputUnsigned()) {
            intMin = 0;
            intMax = APInt::getMaxValue(outBitWidth).getZExtValue();
          }

          auto intMinVal = arith::ConstantOp::create(
              nestedBuilder, loc, nestedBuilder.getI32IntegerAttr(intMin));
          auto intMaxVal = arith::ConstantOp::create(
              nestedBuilder, loc, nestedBuilder.getI32IntegerAttr(intMax));

          value = clampIntHelper(nestedLoc, value, intMinVal, intMaxVal,
                                 nestedBuilder, /*isUnsigned=*/false);

          if (outIntType.getWidth() < 32) {
            value = arith::TruncIOp::create(
                nestedBuilder, nestedLoc,
                rewriter.getIntegerType(outIntType.getWidth()), value);
          }

          if (outIntType.isUnsignedInteger()) {
            value = UnrealizedConversionCastOp::create(nestedBuilder, nestedLoc,
                                                       outIntType, value)
                        .getResult(0);
          }
          linalg::YieldOp::create(nestedBuilder, loc, value);
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
    const bool isBilinear = op.getMode() == ResizeMode::BILINEAR;

    auto inputH = inputTy.getDimSize(1);
    auto inputW = inputTy.getDimSize(2);
    auto outputH = resultTy.getDimSize(1);
    auto outputW = resultTy.getDimSize(2);

    if (inputH != 1 || inputW != 1 || outputH != 1 || outputW != 1)
      return rewriter.notifyMatchFailure(
          op, "tosa.resize is not a pure 1x1->1x1 image operation");

    if (op.getMode() != ResizeMode::NEAREST_NEIGHBOR &&
        op.getMode() != ResizeMode::BILINEAR)
      return rewriter.notifyMatchFailure(
          op, "tosa.resize mode should be NEAREST_NEIGHBOR or BILINEAR");

    if (inputTy == resultTy) {
      rewriter.replaceOp(op, input);
      return success();
    }

    SmallVector<int64_t> scale;
    if (!tosa::getConstShapeValues(op.getScale().getDefiningOp(), scale)) {
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
    Value collapse = tensor::CollapseShapeOp::create(builder, collapseTy, input,
                                                     reassociationMap);

    // Get any dynamic shapes that appear in the input format.
    llvm::SmallVector<Value> outputDynSize;
    if (inputTy.isDynamicDim(0))
      outputDynSize.push_back(tensor::DimOp::create(builder, input, 0));
    if (inputTy.isDynamicDim(3))
      outputDynSize.push_back(tensor::DimOp::create(builder, input, 3));

    // Generate the elementwise operation for casting scaling the input value.
    auto genericTy = collapseTy.clone(resultTy.getElementType());
    Value empty =
        tensor::EmptyOp::create(builder, genericTy.getShape(),
                                resultTy.getElementType(), outputDynSize);
    auto genericMap = rewriter.getMultiDimIdentityMap(genericTy.getRank());
    SmallVector<utils::IteratorType> iterators(genericTy.getRank(),
                                               utils::IteratorType::parallel);

    auto generic = linalg::GenericOp::create(
        builder, genericTy, ValueRange{collapse}, ValueRange{empty},
        ArrayRef<AffineMap>{genericMap, genericMap}, iterators,
        [=](OpBuilder &b, Location loc, ValueRange args) {
          Value value = args[0];
          // This is the quantized case.
          if (inputTy.getElementType() != resultTy.getElementType()) {
            value = arith::ExtSIOp::create(b, loc, resultTy.getElementType(),
                                           value);

            if (isBilinear && scale[0] != 0) {
              Value scaleY = arith::ConstantOp::create(
                  b, loc, b.getI32IntegerAttr(scale[0]));
              value = arith::MulIOp::create(b, loc, value, scaleY);
            }

            if (isBilinear && scale[2] != 0) {
              Value scaleX = arith::ConstantOp::create(
                  b, loc, b.getI32IntegerAttr(scale[2]));
              value = arith::MulIOp::create(b, loc, value, scaleX);
            }
          }

          linalg::YieldOp::create(b, loc, value);
        });

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, resultTy, generic.getResults()[0], reassociationMap);
    return success();
  }
};

// TOSA resize with width or height of 1 may be broadcasted to a wider
// dimension. This is done by materializing a new tosa.resize without
// the broadcasting behavior, and an explicit broadcast afterwards.
class MaterializeResizeBroadcast : public OpRewritePattern<tosa::ResizeOp> {
public:
  using OpRewritePattern<tosa::ResizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ResizeOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder builder(loc, rewriter);
    auto input = op.getInput();
    auto inputTy = dyn_cast<RankedTensorType>(input.getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getType());

    if (!inputTy || !resultTy)
      return rewriter.notifyMatchFailure(op,
                                         "requires ranked input/output types");

    auto batch = inputTy.getDimSize(0);
    auto channels = inputTy.getDimSize(3);
    auto inputH = inputTy.getDimSize(1);
    auto inputW = inputTy.getDimSize(2);
    auto outputH = resultTy.getDimSize(1);
    auto outputW = resultTy.getDimSize(2);

    if ((inputH != 1 || outputH == 1) && (inputW != 1 || outputW == 1))
      return rewriter.notifyMatchFailure(
          op, "tosa.resize has no broadcasting behavior");

    // For any dimension that is broadcastable we generate a width of 1
    // on the output.
    llvm::SmallVector<int64_t> resizeShape;
    resizeShape.push_back(batch);
    resizeShape.push_back(inputH == 1 ? 1 : outputH);
    resizeShape.push_back(inputW == 1 ? 1 : outputW);
    resizeShape.push_back(channels);

    auto resizeTy = resultTy.clone(resizeShape);
    auto resize =
        tosa::ResizeOp::create(builder, resizeTy, input, op.getScale(),
                               op.getOffset(), op.getBorder(), op.getMode());

    // Collapse an unit result dims.
    SmallVector<ReassociationExprs, 4> reassociationMap(2);
    reassociationMap[0].push_back(builder.getAffineDimExpr(0));
    reassociationMap.back().push_back(builder.getAffineDimExpr(1));
    if (inputH != 1)
      reassociationMap.push_back({});
    reassociationMap.back().push_back(builder.getAffineDimExpr(2));
    if (inputW != 1)
      reassociationMap.push_back({});
    reassociationMap.back().push_back(builder.getAffineDimExpr(3));

    llvm::SmallVector<int64_t> collapseShape = {batch};
    if (inputH != 1)
      collapseShape.push_back(outputH);
    if (inputW != 1)
      collapseShape.push_back(outputW);
    collapseShape.push_back(channels);

    auto collapseTy = resultTy.clone(collapseShape);
    Value collapse = tensor::CollapseShapeOp::create(builder, collapseTy,
                                                     resize, reassociationMap);

    // Broadcast the collapsed shape to the output result.
    llvm::SmallVector<Value> outputDynSize;
    if (inputTy.isDynamicDim(0))
      outputDynSize.push_back(tensor::DimOp::create(builder, input, 0));
    if (inputTy.isDynamicDim(3))
      outputDynSize.push_back(tensor::DimOp::create(builder, input, 3));

    SmallVector<utils::IteratorType> iterators(resultTy.getRank(),
                                               utils::IteratorType::parallel);
    Value empty = tensor::EmptyOp::create(
        builder, resultTy.getShape(), resultTy.getElementType(), outputDynSize);

    SmallVector<AffineExpr, 4> inputExprs{rewriter.getAffineDimExpr(0)};
    if (inputH != 1)
      inputExprs.push_back(rewriter.getAffineDimExpr(1));
    if (inputW != 1)
      inputExprs.push_back(rewriter.getAffineDimExpr(2));
    inputExprs.push_back(rewriter.getAffineDimExpr(3));

    auto inputMap = AffineMap::get(resultTy.getRank(), /*symbolCount=*/0,
                                   inputExprs, rewriter.getContext());

    auto outputMap = rewriter.getMultiDimIdentityMap(resultTy.getRank());
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, resultTy, ValueRange{collapse}, ValueRange{empty},
        ArrayRef<AffineMap>{inputMap, outputMap}, iterators,
        [=](OpBuilder &b, Location loc, ValueRange args) {
          Value value = args[0];
          linalg::YieldOp::create(b, loc, value);
        });

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

    bool floatingPointMode = isa<FloatType>(resultETy);
    auto floatTy = resultETy;

    auto imageH = inputTy.getShape()[1];
    auto imageW = inputTy.getShape()[2];

    auto dynamicDimsOr =
        checkHasDynamicBatchDims(rewriter, op, {input, op.getOutput()});
    if (!dynamicDimsOr.has_value())
      return rewriter.notifyMatchFailure(
          op, "unable to get dynamic dimensions of tosa.resize");

    if (op.getMode() != ResizeMode::NEAREST_NEIGHBOR &&
        op.getMode() != ResizeMode::BILINEAR)
      return rewriter.notifyMatchFailure(
          op, "tosa.resize mode should be NEAREST_NEIGHBOR or BILINEAR");

    SmallVector<AffineMap, 2> affineMaps = {
        rewriter.getMultiDimIdentityMap(resultTy.getRank())};
    auto emptyTensor = tensor::EmptyOp::create(b, resultTy.getShape(),
                                               resultETy, *dynamicDimsOr);
    auto genericOp = linalg::GenericOp::create(
        b, resultTy, ValueRange({}), ValueRange{emptyTensor}, affineMaps,
        getNParallelLoopsAttrs(resultTy.getRank()));
    Value resize = genericOp.getResult(0);

    {
      OpBuilder::InsertionGuard regionGuard(b);
      b.createBlock(&genericOp.getRegion(), genericOp.getRegion().end(),
                    TypeRange({resultETy}), loc);
      Value batch = linalg::IndexOp::create(b, 0);
      Value y = linalg::IndexOp::create(b, 1);
      Value x = linalg::IndexOp::create(b, 2);
      Value channel = linalg::IndexOp::create(b, 3);

      Value zeroI32 =
          arith::ConstantOp::create(b, b.getZeroAttr(b.getI32Type()));
      Value zeroFp = arith::ConstantOp::create(b, b.getZeroAttr(floatTy));
      Value hMax =
          arith::ConstantOp::create(b, b.getI32IntegerAttr(imageH - 1));
      Value wMax =
          arith::ConstantOp::create(b, b.getI32IntegerAttr(imageW - 1));

      Value inY = arith::IndexCastOp::create(b, b.getI32Type(), y);
      Value inX = arith::IndexCastOp::create(b, b.getI32Type(), x);

      SmallVector<int64_t> scale, offset, border;
      if (!tosa::getConstShapeValues(op.getScale().getDefiningOp(), scale) ||
          !tosa::getConstShapeValues(op.getOffset().getDefiningOp(), offset) ||
          !tosa::getConstShapeValues(op.getBorder().getDefiningOp(), border)) {
        return rewriter.notifyMatchFailure(
            op, "tosa.resize scale/offset/border should have compile time "
                "constant values.");
      }

      Value yScaleN, yScaleD, xScaleN, xScaleD;
      yScaleN = arith::ConstantOp::create(b, b.getI32IntegerAttr(scale[0]));
      yScaleD = arith::ConstantOp::create(b, b.getI32IntegerAttr(scale[1]));
      xScaleN = arith::ConstantOp::create(b, b.getI32IntegerAttr(scale[2]));
      xScaleD = arith::ConstantOp::create(b, b.getI32IntegerAttr(scale[3]));

      Value yOffset, xOffset, yBorder, xBorder;
      yOffset = arith::ConstantOp::create(b, b.getI32IntegerAttr(offset[0]));
      xOffset = arith::ConstantOp::create(b, b.getI32IntegerAttr(offset[1]));
      yBorder = arith::ConstantOp::create(b, b.getI32IntegerAttr(border[0]));
      xBorder = arith::ConstantOp::create(b, b.getI32IntegerAttr(border[1]));

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
        Value val = arith::MulIOp::create(b, in, scaleD);
        val = arith::AddIOp::create(b, val, offset);
        index = arith::FloorDivSIOp::create(b, val, scaleN);

        // rx = x % scale_n
        // dx = rx / scale_n
        Value r = arith::RemSIOp::create(b, val, scaleN);
        Value rFp = arith::SIToFPOp::create(b, floatTy, r);
        Value scaleNfp = arith::UIToFPOp::create(b, floatTy, scaleN);
        delta = arith::DivFOp::create(b, rFp, scaleNfp);
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
        Value val = arith::MulIOp::create(b, in, scaleD);
        val = arith::AddIOp::create(b, val, offset);
        index = arith::DivSIOp::create(b, val, scaleN);
        delta = arith::MulIOp::create(b, index, scaleN);
        delta = arith::SubIOp::create(b, val, delta);
      };

      Value ix, iy, dx, dy;
      if (floatingPointMode) {
        getIndexAndDeltaFp(iy, dy, inY, yScaleN, yScaleD, yOffset, imageH, b);
        getIndexAndDeltaFp(ix, dx, inX, xScaleN, xScaleD, xOffset, imageW, b);
      } else {
        getIndexAndDeltaInt(iy, dy, inY, yScaleN, yScaleD, yOffset, imageH, b);
        getIndexAndDeltaInt(ix, dx, inX, xScaleN, xScaleD, xOffset, imageW, b);
      }

      if (op.getMode() == ResizeMode::NEAREST_NEIGHBOR) {
        auto one = arith::ConstantOp::create(b, b.getI32IntegerAttr(1));

        auto getNearestIndexAndClamp = [&](Value val, Value dval, Value scale,
                                           Value max, int size,
                                           ImplicitLocOpBuilder &b) -> Value {
          if (size == 1) {
            return arith::ConstantIndexOp::create(b, 0);
          }

          Value pred;
          if (floatingPointMode) {
            auto h =
                arith::ConstantOp::create(b, b.getFloatAttr(floatTy, 0.5f));
            pred = arith::CmpFOp::create(b, arith::CmpFPredicate::OGE, dval, h);
          } else {
            Value dvalDouble = arith::ShLIOp::create(b, dval, one);
            pred = arith::CmpIOp::create(b, arith::CmpIPredicate::sge,
                                         dvalDouble, scale);
          }

          auto offset = arith::SelectOp::create(b, pred, one, zeroI32);
          val = arith::AddIOp::create(b, val, offset);
          val = clampIntHelper(loc, val, zeroI32, max, b, /*isUnsigned=*/false);
          return arith::IndexCastOp::create(b, b.getIndexType(), val);
        };

        iy = getNearestIndexAndClamp(iy, dy, yScaleN, hMax, imageH, b);
        ix = getNearestIndexAndClamp(ix, dx, xScaleN, wMax, imageW, b);

        Value result = tensor::ExtractOp::create(
            b, input, ValueRange{batch, iy, ix, channel});

        linalg::YieldOp::create(b, result);
      } else {
        // The mode here must be BILINEAR.
        assert(op.getMode() == ResizeMode::BILINEAR);

        auto oneVal = arith::ConstantOp::create(b, b.getI32IntegerAttr(1));

        auto getClampedIdxs = [&](Value &val0, Value &val1, int size, Value in,
                                  Value max, ImplicitLocOpBuilder &b) {
          val0 = in;
          val1 = arith::AddIOp::create(b, val0, oneVal);
          val0 =
              clampIntHelper(loc, val0, zeroI32, max, b, /*isUnsigned=*/false);
          val1 =
              clampIntHelper(loc, val1, zeroI32, max, b, /*isUnsigned=*/false);
          val0 = arith::IndexCastOp::create(b, b.getIndexType(), val0);
          val1 = arith::IndexCastOp::create(b, b.getIndexType(), val1);
        };

        // Linalg equivalent to the section below:
        //    int16_t iy0 = apply_max(iy, 0);
        //    int16_t iy1 = apply_min(iy + 1, IH - 1);
        //    int16_t ix0 = apply_max(ix, 0);
        //    int16_t ix1 = apply_min(ix + 1, IW - 1);
        Value x0, x1, y0, y1;
        getClampedIdxs(y0, y1, imageH, iy, hMax, b);
        getClampedIdxs(x0, x1, imageW, ix, wMax, b);

        Value y0x0 = tensor::ExtractOp::create(
            b, input, ValueRange{batch, y0, x0, channel});
        Value y0x1 = tensor::ExtractOp::create(
            b, input, ValueRange{batch, y0, x1, channel});
        Value y1x0 = tensor::ExtractOp::create(
            b, input, ValueRange{batch, y1, x0, channel});
        Value y1x1 = tensor::ExtractOp::create(
            b, input, ValueRange{batch, y1, x1, channel});

        if (floatingPointMode) {
          auto oneVal =
              arith::ConstantOp::create(b, b.getFloatAttr(floatTy, 1.0f));
          auto interpolate = [&](Value val0, Value val1, Value delta,
                                 int inputSize,
                                 ImplicitLocOpBuilder &b) -> Value {
            if (inputSize == 1)
              return val0;
            Value oneMinusDelta = arith::SubFOp::create(b, oneVal, delta);
            Value mul0 = arith::MulFOp::create(b, val0, oneMinusDelta);
            Value mul1 = arith::MulFOp::create(b, val1, delta);
            return arith::AddFOp::create(b, mul0, mul1);
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
          linalg::YieldOp::create(b, result);
        } else {
          // Perform in quantized space.
          y0x0 = arith::ExtSIOp::create(b, resultETy, y0x0);
          y0x1 = arith::ExtSIOp::create(b, resultETy, y0x1);
          y1x0 = arith::ExtSIOp::create(b, resultETy, y1x0);
          y1x1 = arith::ExtSIOp::create(b, resultETy, y1x1);

          const int64_t deltaBitwidth = dx.getType().getIntOrFloatBitWidth();
          if (resultETy.getIntOrFloatBitWidth() > deltaBitwidth) {
            dx = arith::ExtSIOp::create(b, resultETy, dx);
            dy = arith::ExtSIOp::create(b, resultETy, dy);
          }

          Value yScaleNExt = yScaleN;
          Value xScaleNExt = xScaleN;

          const int64_t scaleBitwidth =
              xScaleN.getType().getIntOrFloatBitWidth();
          if (resultETy.getIntOrFloatBitWidth() > scaleBitwidth) {
            yScaleNExt = arith::ExtSIOp::create(b, resultETy, yScaleN);
            xScaleNExt = arith::ExtSIOp::create(b, resultETy, xScaleN);
          }

          auto interpolate = [](Value val0, Value val1, Value weight1,
                                Value scale, int inputSize,
                                ImplicitLocOpBuilder &b) -> Value {
            if (inputSize == 1)
              return arith::MulIOp::create(b, val0, scale);
            Value weight0 = arith::SubIOp::create(b, scale, weight1);
            Value mul0 = arith::MulIOp::create(b, val0, weight0);
            Value mul1 = arith::MulIOp::create(b, val1, weight1);
            return arith::AddIOp::create(b, mul0, mul1);
          };

          Value topAcc = interpolate(y0x0, y0x1, dx, xScaleNExt, imageW, b);
          Value bottomAcc = interpolate(y1x0, y1x1, dx, xScaleNExt, imageW, b);
          Value result =
              interpolate(topAcc, bottomAcc, dy, yScaleNExt, imageH, b);
          linalg::YieldOp::create(b, result);
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
        dynDims.push_back(tensor::DimOp::create(rewriter, loc, input, i));
      }
    }

    Value axisDimSize = tensor::DimOp::create(rewriter, loc, input, axis);

    // First fill the output buffer with the init value.
    auto emptyTensor = tensor::EmptyOp::create(
                           rewriter, loc, inputTy.getShape(),
                           inputTy.getElementType(), ArrayRef<Value>({dynDims}))
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
                linalg::IndexOp::create(rewriter, nestedLoc, i).getResult();
            if (i == axis) {
              auto one = arith::ConstantIndexOp::create(rewriter, nestedLoc, 1);
              auto sizeMinusOne =
                  arith::SubIOp::create(rewriter, nestedLoc, axisDimSize, one);
              index = arith::SubIOp::create(rewriter, nestedLoc, sizeMinusOne,
                                            index);
            }

            indices.push_back(index);
          }

          auto extract = tensor::ExtractOp::create(nestedBuilder, nestedLoc,
                                                   input, indices);
          linalg::YieldOp::create(nestedBuilder, op.getLoc(),
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
        dynDims.push_back(tensor::DimOp::create(rewriter, loc, input, i));
      }
    }

    auto emptyTensor = tensor::EmptyOp::create(
        rewriter, op.getLoc(), genericShape, elementTy, dynDims);

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

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, RankedTensorType::get(genericShape, elementTy), input,
        ValueRange{emptyTensor}, affineMaps,
        getNParallelLoopsAttrs(genericShape.size()),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          linalg::YieldOp::create(nestedBuilder, op.getLoc(), *args.begin());
        });

    auto shapeValue = getTosaConstShape(
        rewriter, loc, mlir::tosa::convertFromMlirShape(resultTy.getShape()));
    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
        op, resultTy, genericOp.getResult(0), shapeValue);
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
        dynDims.push_back(tensor::DimOp::create(rewriter, loc, input, i));
      }
    }

    // First fill the output buffer for the index.
    auto emptyTensorIdx =
        tensor::EmptyOp::create(rewriter, loc, resultTy.getShape(),
                                outElementTy, dynDims)
            .getResult();
    auto fillValueIdx = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(outElementTy, 0));
    auto filledTensorIdx =
        linalg::FillOp::create(rewriter, loc, ValueRange{fillValueIdx},
                               ValueRange{emptyTensorIdx})
            .result();

    // Second fill the output buffer for the running max.
    auto emptyTensorMax =
        tensor::EmptyOp::create(rewriter, loc, resultTy.getShape(), inElementTy,
                                dynDims)
            .getResult();
    auto fillValueMaxAttr =
        createInitialValueForReduceOp(argmaxOp, inElementTy, rewriter);

    if (!fillValueMaxAttr)
      return rewriter.notifyMatchFailure(
          argmaxOp, "unsupported tosa.argmax element type");

    auto fillValueMax =
        arith::ConstantOp::create(rewriter, loc, fillValueMaxAttr);
    auto filledTensorMax =
        linalg::FillOp::create(rewriter, loc, ValueRange{fillValueMax},
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
    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, ArrayRef<Type>({resultTy, resultMaxTy}), input,
        ValueRange({filledTensorIdx, filledTensorMax}), maps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          auto newValue = blockArgs[0];
          auto oldIndex = blockArgs[1];
          auto oldValue = blockArgs[2];

          Value newIndex = arith::IndexCastOp::create(
              rewriter, nestedLoc, oldIndex.getType(),
              linalg::IndexOp::create(rewriter, loc, axis));

          Value predicate;
          if (isa<FloatType>(inElementTy)) {
            if (argmaxOp.getNanMode() == NanPropagationMode::IGNORE) {
              // Only update index & max value for non NaN values. If all
              // values are NaNs, the initial index will be return which is 0.
              predicate = arith::CmpFOp::create(rewriter, nestedLoc,
                                                arith::CmpFPredicate::OGT,
                                                newValue, oldValue);
            } else {
              // Update max value if either of the following is true:
              // - new value is bigger
              // - cur max is not NaN and new value is NaN
              Value gt = arith::CmpFOp::create(rewriter, nestedLoc,
                                               arith::CmpFPredicate::UGT,
                                               newValue, oldValue);
              Value oldNonNaN = arith::CmpFOp::create(rewriter, nestedLoc,
                                                      arith::CmpFPredicate::ORD,
                                                      oldValue, oldValue);
              predicate = arith::AndIOp::create(
                  rewriter, nestedLoc, rewriter.getI1Type(), gt, oldNonNaN);
            }
          } else if (isa<IntegerType>(inElementTy)) {
            predicate = arith::CmpIOp::create(rewriter, nestedLoc,
                                              arith::CmpIPredicate::sgt,
                                              newValue, oldValue);
          } else {
            didEncounterError = true;
            return;
          }

          auto resultMax = arith::SelectOp::create(
              rewriter, nestedLoc, predicate, newValue, oldValue);
          auto resultIndex = arith::SelectOp::create(
              rewriter, nestedLoc, predicate, newIndex, oldIndex);
          linalg::YieldOp::create(nestedBuilder, nestedLoc,
                                  ValueRange({resultIndex, resultMax}));
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

    auto valuesTy = dyn_cast<RankedTensorType>(op.getValues().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getType());
    if (!valuesTy || !resultTy)
      return rewriter.notifyMatchFailure(op, "unranked tensors not supported");

    auto dynamicDims = inferDynamicDimsForGather(
        rewriter, op.getLoc(), adaptor.getValues(), adaptor.getIndices());

    auto resultElementTy = resultTy.getElementType();

    auto loc = op.getLoc();
    auto emptyTensor =
        tensor::EmptyOp::create(rewriter, loc, resultTy.getShape(),
                                resultElementTy, dynamicDims)
            .getResult();

    SmallVector<AffineMap, 2> affineMaps = {
        AffineMap::get(
            /*dimCount=*/resultTy.getRank(), /*symbolCount=*/0,
            {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)},
            rewriter.getContext()),
        rewriter.getMultiDimIdentityMap(resultTy.getRank())};

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, ArrayRef<Type>({resultTy}), ValueRange{indices},
        ValueRange{emptyTensor}, affineMaps,
        getNParallelLoopsAttrs(resultTy.getRank()),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto indexValue = args[0];
          auto index0 = linalg::IndexOp::create(rewriter, loc, 0);
          Value index1 = arith::IndexCastOp::create(
              rewriter, loc, rewriter.getIndexType(), indexValue);
          auto index2 = linalg::IndexOp::create(rewriter, loc, 2);
          Value extract = tensor::ExtractOp::create(
              rewriter, loc, input, ValueRange{index0, index1, index2});
          linalg::YieldOp::create(rewriter, loc, extract);
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
            tensor::DimOp::create(rewriter, loc, op.getOperand(0), i));
      }
    }

    auto emptyTensor =
        tensor::EmptyOp::create(rewriter, loc, resultTy.getShape(),
                                resultElementTy, dynDims)
            .getResult();

    SmallVector<AffineMap, 2> affineMaps = {
        rewriter.getMultiDimIdentityMap(resultTy.getRank()),
        rewriter.getMultiDimIdentityMap(resultTy.getRank())};

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, resultTy, ValueRange({input}), ValueRange{emptyTensor},
        affineMaps, getNParallelLoopsAttrs(resultTy.getRank()));
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
        Value index = arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), inputValue);
        Value offset = arith::ConstantIndexOp::create(rewriter, loc, 128);
        index = arith::AddIOp::create(rewriter, loc, rewriter.getIndexType(),
                                      index, offset);
        Value extract =
            tensor::ExtractOp::create(rewriter, loc, table, ValueRange{index});
        linalg::YieldOp::create(rewriter, loc, extract);
        return success();
      }

      if (inputElementTy.isInteger(16) && tableElementTy.isInteger(16) &&
          resultElementTy.isInteger(32)) {
        Value extend = arith::ExtSIOp::create(
            rewriter, loc, rewriter.getI32Type(), inputValue);

        auto offset = arith::ConstantOp::create(
            rewriter, loc, rewriter.getI32IntegerAttr(32768));
        auto seven = arith::ConstantOp::create(rewriter, loc,
                                               rewriter.getI32IntegerAttr(7));
        auto one = arith::ConstantOp::create(rewriter, loc,
                                             rewriter.getI32IntegerAttr(1));
        auto b1111111 = arith::ConstantOp::create(
            rewriter, loc, rewriter.getI32IntegerAttr(127));

        // Compute the index and fractional part from the input value:
        // value = value + 32768
        // index = value >> 7;
        // fraction = 0x01111111 & value
        auto extendAdd = arith::AddIOp::create(rewriter, loc, extend, offset);
        Value index = arith::ShRUIOp::create(rewriter, loc, extendAdd, seven);
        Value fraction =
            arith::AndIOp::create(rewriter, loc, extendAdd, b1111111);

        // Extract the base and next values from the table.
        // base = (int32_t) table[index];
        // next = (int32_t) table[index + 1];
        Value indexPlusOne = arith::AddIOp::create(rewriter, loc, index, one);

        index = arith::IndexCastOp::create(rewriter, loc,
                                           rewriter.getIndexType(), index);
        indexPlusOne = arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), indexPlusOne);

        Value base =
            tensor::ExtractOp::create(rewriter, loc, table, ValueRange{index});
        Value next = tensor::ExtractOp::create(rewriter, loc, table,
                                               ValueRange{indexPlusOne});

        base =
            arith::ExtSIOp::create(rewriter, loc, rewriter.getI32Type(), base);
        next =
            arith::ExtSIOp::create(rewriter, loc, rewriter.getI32Type(), next);

        // Use the fractional part to interpolate between the input values:
        // result = (base << 7) + (next - base) * fraction
        Value baseScaled = arith::ShLIOp::create(rewriter, loc, base, seven);
        Value diff = arith::SubIOp::create(rewriter, loc, next, base);
        Value diffScaled = arith::MulIOp::create(rewriter, loc, diff, fraction);
        Value result =
            arith::AddIOp::create(rewriter, loc, baseScaled, diffScaled);

        linalg::YieldOp::create(rewriter, loc, result);

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
    auto one = arith::ConstantIndexOp::create(builder, loc, 1);
    auto two = arith::ConstantIndexOp::create(builder, loc, 2);

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
        tensor::EmptyOp::create(rewriter, loc, type, dynamicSizes);
    auto fillValueAttr = rewriter.getZeroAttr(type.getElementType());
    auto fillValue = arith::ConstantOp::create(rewriter, loc, fillValueAttr);
    auto filledTensor =
        linalg::FillOp::create(rewriter, loc, ValueRange{fillValue},
                               ValueRange{emptyTensor})
            .result();
    return filledTensor;
  }

  static Value castIndexToFloat(OpBuilder &builder, Location loc,
                                FloatType type, Value value) {
    auto integerVal = arith::IndexCastUIOp::create(
        builder, loc,
        type.getIntOrFloatBitWidth() > 32 ? builder.getI64Type()
                                          : builder.getI32Type(),
        value);

    return arith::UIToFPOp::create(builder, loc, type, integerVal);
  }

  static Value createLinalgIndex(OpBuilder &builder, Location loc,
                                 FloatType type, int64_t index) {
    auto indexVal = linalg::IndexOp::create(builder, loc, index);
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
    auto input = rfft2d.getInputReal();
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
    auto twoPi = arith::ConstantOp::create(rewriter, loc, twoPiAttr);
    auto constH = castIndexToFloat(rewriter, loc, elementType, dimH);
    auto constW = castIndexToFloat(rewriter, loc, elementType, dimW);

    auto buildBody = [&](OpBuilder &builder, Location loc, ValueRange args) {
      Value valReal = args[0];
      Value sumReal = args[1];
      Value sumImag = args[2];

      // Indices for angle computation
      Value oy = linalg::IndexOp::create(builder, loc, 1);
      Value ox = linalg::IndexOp::create(builder, loc, 2);
      Value iy = linalg::IndexOp::create(builder, loc, 3);
      Value ix = linalg::IndexOp::create(builder, loc, 4);

      // Calculating angle without integer parts of components as sin/cos are
      // periodic: angle = 2 * pi() * ( ( (iy * oy) % H) / H + ( (ix * ox) % W )
      // / W);
      auto iyXoy = index::MulOp::create(builder, loc, iy, oy);
      auto ixXox = index::MulOp::create(builder, loc, ix, ox);

      auto iyRem = index::RemUOp::create(builder, loc, iyXoy, dimH);
      auto ixRem = index::RemUOp::create(builder, loc, ixXox, dimW);

      auto iyRemFloat = castIndexToFloat(builder, loc, elementType, iyRem);
      auto ixRemFloat = castIndexToFloat(builder, loc, elementType, ixRem);

      auto yComponent = arith::DivFOp::create(builder, loc, iyRemFloat, constH);
      auto xComponent = arith::DivFOp::create(builder, loc, ixRemFloat, constW);
      auto sumXY = arith::AddFOp::create(builder, loc, yComponent, xComponent);
      auto angle = arith::MulFOp::create(builder, loc, twoPi, sumXY);

      // realComponent = valReal * cos(angle)
      // imagComponent = valReal * sin(angle)
      auto cosAngle = math::CosOp::create(builder, loc, angle);
      auto sinAngle = math::SinOp::create(builder, loc, angle);
      auto realComponent =
          arith::MulFOp::create(builder, loc, valReal, cosAngle);
      auto imagComponent =
          arith::MulFOp::create(builder, loc, valReal, sinAngle);

      // outReal = sumReal + realComponent
      // outImag = sumImag - imagComponent
      auto outReal =
          arith::AddFOp::create(builder, loc, sumReal, realComponent);
      auto outImag =
          arith::SubFOp::create(builder, loc, sumImag, imagComponent);

      linalg::YieldOp::create(builder, loc, ValueRange{outReal, outImag});
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
    auto twoPi = arith::ConstantOp::create(rewriter, loc, twoPiAttr);
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
      Value oy = linalg::IndexOp::create(builder, loc, 1);
      Value ox = linalg::IndexOp::create(builder, loc, 2);
      Value iy = linalg::IndexOp::create(builder, loc, 3);
      Value ix = linalg::IndexOp::create(builder, loc, 4);

      // float_t angle = sign_val * 2 * pi() * ( ( (iy * oy) % H) / H + ( (ix *
      // ox) % W ) / W);
      auto iyXoy = index::MulOp::create(builder, loc, iy, oy);
      auto ixXox = index::MulOp::create(builder, loc, ix, ox);

      auto iyRem = index::RemUOp::create(builder, loc, iyXoy, dimH);
      auto ixRem = index::RemUOp::create(builder, loc, ixXox, dimW);

      auto iyRemFloat =
          RFFT2dConverter::castIndexToFloat(builder, loc, real_el_ty, iyRem);
      auto ixRemFloat =
          RFFT2dConverter::castIndexToFloat(builder, loc, real_el_ty, ixRem);

      auto yComponent = arith::DivFOp::create(builder, loc, iyRemFloat, constH);
      auto xComponent = arith::DivFOp::create(builder, loc, ixRemFloat, constW);

      auto sumXY = arith::AddFOp::create(builder, loc, yComponent, xComponent);
      auto angle = arith::MulFOp::create(builder, loc, twoPi, sumXY);

      if (inverse.getValue()) {
        angle = arith::MulFOp::create(
            builder, loc, angle,
            arith::ConstantOp::create(rewriter, loc,
                                      rewriter.getFloatAttr(real_el_ty, -1.0)));
      }

      // realComponent = val_real * cos(a) + val_imag * sin(a);
      // imagComponent = -val_real * sin(a) + val_imag * cos(a);
      auto cosAngle = math::CosOp::create(builder, loc, angle);
      auto sinAngle = math::SinOp::create(builder, loc, angle);

      auto rcos = arith::MulFOp::create(builder, loc, valReal, cosAngle);
      auto rsin = arith::MulFOp::create(builder, loc, valImag, sinAngle);
      auto realComponent = arith::AddFOp::create(builder, loc, rcos, rsin);

      auto icos = arith::MulFOp::create(builder, loc, valImag, cosAngle);
      auto isin = arith::MulFOp::create(builder, loc, valReal, sinAngle);

      auto imagComponent = arith::SubFOp::create(builder, loc, icos, isin);

      // outReal = sumReal + realComponent
      // outImag = sumImag - imagComponent
      auto outReal =
          arith::AddFOp::create(builder, loc, sumReal, realComponent);
      auto outImag =
          arith::AddFOp::create(builder, loc, sumImag, imagComponent);

      linalg::YieldOp::create(builder, loc, ValueRange{outReal, outImag});
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
  patterns->add<MaterializeResizeBroadcast>(patterns->getContext(),
                                            /*benefit=*/300);

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
      ReduceConverter<tosa::ReduceProductOp>,
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
