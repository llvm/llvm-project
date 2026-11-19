//===- ExpandOps.cpp - Pass to legalize Arith ops for LLVM lowering --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithCommon/CeilFloorDivExpansion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Utils/NarrowFloatExpansion.h"
#include "mlir/Dialect/Vector/IR/VectorDialect.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace arith {
#define GEN_PASS_DEF_ARITHEXPANDOPSPASS
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace arith
} // namespace mlir

using namespace mlir;

/// Create an integer or index constant.
static Value createConst(Location loc, Type type, int value,
                         PatternRewriter &rewriter) {
  auto attr = rewriter.getIntegerAttr(getElementTypeOrSelf(type), value);
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    return arith::ConstantOp::create(rewriter, loc,
                                     DenseElementsAttr::get(shapedTy, attr));
  }
  return arith::ConstantOp::create(rewriter, loc, attr);
}

/// Create an integer constant from an APInt.
static Value createAPIntConst(Location loc, Type type, const APInt &value,
                              PatternRewriter &rewriter) {
  auto attr = IntegerAttr::get(getElementTypeOrSelf(type), value);
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    return arith::ConstantOp::create(rewriter, loc,
                                     DenseElementsAttr::get(shapedTy, attr));
  }
  return arith::ConstantOp::create(rewriter, loc, attr);
}

/// Creates shapedType using shape from cloneFrom and base type from cloneTo
static Type cloneToShapedType(Type cloneFrom, Type cloneTo) {
  if (auto shapedTy = dyn_cast<ShapedType>(cloneFrom)) {
    return shapedTy.clone(cloneTo);
  }
  return cloneTo;
}

/// Returns the exact f32 bits of the f8E8M0FNU `code`: code << 23, except 0x00
/// (2^-127, 0x00400000) and, if `withNaN`, the NaN (0x7FC00000).
static Value emitF8E8M0FNUToF32Bits(ImplicitLocOpBuilder &b,
                                    PatternRewriter &rewriter, Value code,
                                    bool withNaN) {
  Location loc = b.getLoc();
  Type i8Ty = code.getType();
  Type i32Ty = cloneToShapedType(i8Ty, b.getI32Type());
  Value cF32MantissaWidth = createConst(loc, i32Ty, 23, rewriter);
  Value exti = arith::ExtUIOp::create(b, i32Ty, code);
  Value f32Bits = arith::ShLIOp::create(b, exti, cF32MantissaWidth);
  Value cF8Min = createConst(loc, i8Ty, 0, rewriter);
  Value cF32Min = createConst(loc, i32Ty, 0x00400000, rewriter);
  Value min = arith::CmpIOp::create(b, arith::CmpIPredicate::eq, code, cF8Min);
  f32Bits = arith::SelectOp::create(b, min, cF32Min, f32Bits);
  if (!withNaN)
    return f32Bits;
  Value cF8NaN = createConst(loc, i8Ty, 0xff, rewriter);
  Value cF32NaN = createConst(loc, i32Ty, 0x7fc00000, rewriter);
  Value isNan =
      arith::CmpIOp::create(b, arith::CmpIPredicate::eq, code, cF8NaN);
  return arith::SelectOp::create(b, isNan, cF32NaN, f32Bits);
}

static bool isEmitterWideType(Type ty) {
  return ty.isF16() || ty.isBF16() || ty.isF32() || ty.isF64();
}

/// A float the emitters take: at most 8 bits, with a zero and subnormals or,
/// as a truncation destination only, an unsigned precision-1 format with
/// neither.
static bool isNarrowFloat(Type ty, bool asTruncDest = false) {
  auto floatTy = dyn_cast<FloatType>(ty);
  if (!floatTy || floatTy.getWidth() > 8)
    return false;
  const llvm::fltSemantics &sem = floatTy.getFloatSemantics();
  if (sem.hasZero && sem.hasDenormals)
    return true;
  return asTruncDest && !sem.hasZero && !sem.hasDenormals &&
         !sem.hasSignedRepr && APFloat::semanticsPrecision(sem) == 1;
}

/// Whether every finite `from` value is exact in `to` (or, if `allowOverflow`,
/// above its range); given precision and sign, the extremes decide.
static bool isExactIn(FloatType from, FloatType to,
                      bool allowOverflow = false) {
  const llvm::fltSemantics &fromSem = from.getFloatSemantics();
  const llvm::fltSemantics &toSem = to.getFloatSemantics();
  if (APFloat::semanticsPrecision(fromSem) >
          APFloat::semanticsPrecision(toSem) ||
      (APFloat::semanticsHasSignedRepr(fromSem) &&
       !APFloat::semanticsHasSignedRepr(toSem)))
    return false;
  SmallVector<APFloat, 2> values = {APFloat::getSmallest(fromSem)};
  if (!allowOverflow)
    values.push_back(APFloat::getLargest(fromSem));
  for (APFloat &value : values) {
    bool losesInfo = false;
    (void)value.convert(toSem, APFloat::rmNearestTiesToEven, &losesInfo);
    if (losesInfo)
      return false;
  }
  return true;
}

static bool isExactNarrowPair(FloatType src, FloatType dst) {
  return isNarrowFloat(dst) && isExactIn(src, dst);
}

/// Whether emitNarrowFloatTrunc's exponent-range and carrier asserts hold.
static bool canRoundFrom(FloatType src, FloatType dst) {
  const llvm::fltSemantics &srcSem = src.getFloatSemantics();
  const llvm::fltSemantics &dstSem = dst.getFloatSemantics();
  return APFloat::semanticsMinExponent(dstSem) >=
             APFloat::semanticsMinExponent(srcSem) &&
         APFloat::semanticsMaxExponent(dstSem) <=
             APFloat::semanticsMaxExponent(srcSem) &&
         APFloat::semanticsPrecision(srcSem) +
                 APFloat::semanticsPrecision(dstSem) + 2 <=
             64;
}

/// Returns the type emitNarrowFloatTrunc rounds from: `src`; f64 for f80 and
/// f128 (see emitRoundToOddF64Bits); else f32, or f64 if f32's range is too
/// small, reached exactly (extf, or a decode); or null.
static FloatType getTruncSourceType(FloatType src, FloatType dst) {
  if (isEmitterWideType(src) && canRoundFrom(src, dst))
    return src;
  FloatType f64 = Float64Type::get(src.getContext());
  if (isa<Float80Type, Float128Type>(src))
    return canRoundFrom(f64, dst) ? f64 : FloatType();
  FloatType f32 = Float32Type::get(src.getContext());
  if (isa<Float8E8M0FNUType>(src))
    return canRoundFrom(f32, dst) ? f32 : FloatType();
  FloatType wide = canRoundFrom(f32, dst) ? f32 : f64;
  if (!canRoundFrom(wide, dst))
    return FloatType();
  bool decodable = isNarrowFloat(src);
  bool extendable = src.getWidth() > 8 && src.getWidth() < wide.getWidth();
  return (decodable || extendable) && isExactIn(src, wide) ? wide : FloatType();
}

/// Returns the type emitNarrowFloatExt writes for `src` to `dst`, or null: f16
/// if `src` has its exponent layout and `dst` is wider; else `dst`; f32 (then a
/// truncf) for a `dst` of at most 8 bits without a zero, subnormals or `src`'s
/// sign; else the widest of f64, f32 and f16 holding `src`, exact in `dst`.
static FloatType getExtResultType(FloatType src, FloatType dst) {
  Builder builder(dst.getContext());
  FloatType f16 = builder.getF16Type();
  if (dst.getWidth() > 16 && isExactIn(f16, dst) &&
      arith::haveSameExponentLayout(arith::floatFormatSpecFromFloatType(src),
                                    arith::floatFormatSpecFromFloatType(f16)))
    return f16;
  if (isEmitterWideType(dst))
    return isExactIn(src, dst, /*allowOverflow=*/true) ? dst : FloatType();
  if (dst.getWidth() <= 8) {
    bool losesSign = APFloat::semanticsHasSignedRepr(src.getFloatSemantics()) &&
                     !APFloat::semanticsHasSignedRepr(dst.getFloatSemantics());
    FloatType f32 = builder.getF32Type();
    return (!isNarrowFloat(dst) || losesSign) && isExactIn(src, f32)
               ? f32
               : FloatType();
  }
  for (FloatType wide :
       {builder.getF64Type(), builder.getF32Type(), builder.getF16Type()})
    if (wide.getWidth() < dst.getWidth() && isExactIn(wide, dst) &&
        isExactIn(src, wide))
      return wide;
  return FloatType();
}

/// The gate shared by the converters and the pass legality. A truncf into a
/// listed type (which may lack a zero), or an exact extf between narrow floats
/// one of them listed, is a rounding; any other extf from a listed type an
/// extension. Dynamic shapes are left alone (no splat constants).
static bool canExpandNarrowFloat(bool isExt, Type srcTy, Type dstTy, Type opTy,
                                 ArrayRef<TypeID> narrowTypes) {
  if (auto shapedTy = dyn_cast<ShapedType>(opTy);
      shapedTy && !shapedTy.hasStaticShape())
    return false;
  auto src = dyn_cast<FloatType>(srcTy);
  auto dst = dyn_cast<FloatType>(dstTy);
  if (!src || !dst)
    return false;
  auto isListed = [&](FloatType ty, bool asTruncDest = false) {
    return isNarrowFloat(ty, asTruncDest) &&
           llvm::is_contained(narrowTypes, ty.getTypeID());
  };
  if (!isExt)
    return isListed(dst, /*asTruncDest=*/true) && getTruncSourceType(src, dst);
  if (isExactNarrowPair(src, dst))
    return (isListed(dst) || isListed(src)) && getTruncSourceType(src, dst);
  return isListed(src) && getExtResultType(src, dst);
}

/// i32, or i64 if a format or the truncation's rounding needs more bits.
static IntegerType getCarrierIntType(Builder &b,
                                     const arith::FloatFormatSpec &src,
                                     const arith::FloatFormatSpec &dst) {
  bool needsI64 = src.sizeInBits > 32 || dst.sizeInBits > 32 ||
                  src.mantBits + dst.mantBits + 4 > 32;
  return b.getIntegerType(needsI64 ? 64 : 32);
}

/// Returns the op's nnan flag alone, or null: the only flag of the float ops
/// created before the rounding. The f64 value of an f128 can be Inf or -0
/// where the f128 is neither (ninf, nsz), and with contract ExtFOp::fold would
/// fold the widening of a contract truncf back to the truncf's operand.
static arith::FastMathFlagsAttr getNNaNOnly(MLIRContext *ctx,
                                            arith::FastMathFlagsAttr fastmath) {
  if (fastmath && arith::bitEnumContainsAll(fastmath.getValue(),
                                            arith::FastMathFlags::nnan))
    return arith::FastMathFlagsAttr::get(ctx, arith::FastMathFlags::nnan);
  return {};
}

/// Returns the f64 bits of the f128 or f80 `operand` rounded to odd: toward
/// zero, with the LSB set if inexact. f64 has at least two more bits than any
/// precision p the emitter writes (53 >= p + 2) and a wider range, so rounding
/// this value once more in any mode equals rounding `operand` directly. An f80
/// is extended to f128 first (exact for canonical x87 encodings; a
/// non-canonical one is read as the lowered arith.extf reads it). A NaN gives
/// APFloat's f64 NaN; `nnan` (the op's nnan flag or null) drops that step.
static Value emitRoundToOddF64Bits(ImplicitLocOpBuilder &b,
                                   PatternRewriter &rewriter, Value operand,
                                   arith::FastMathFlagsAttr nnan) {
  Location loc = b.getLoc();
  Type operandTy = operand.getType();
  Type f128Ty = cloneToShapedType(operandTy, b.getF128Type());
  Type i128Ty = cloneToShapedType(operandTy, b.getIntegerType(128));
  Type f64Ty = cloneToShapedType(operandTy, b.getF64Type());
  Type i64Ty = cloneToShapedType(operandTy, b.getI64Type());
  Value x = operand;
  if (operandTy != f128Ty)
    x = arith::ExtFOp::create(b, f128Ty, operand, nnan);
  Value r =
      arith::TruncFOp::create(b, f64Ty, x, /*roundingmode=*/nullptr, nnan);
  Value back = arith::ExtFOp::create(b, f128Ty, r, nnan);
  // `back` has the sign of `x`, so their bits order by magnitude.
  Value xBits = arith::BitcastOp::create(b, i128Ty, x);
  Value backBits = arith::BitcastOp::create(b, i128Ty, back);
  Value inexact =
      arith::CmpIOp::create(b, arith::CmpIPredicate::ne, backBits, xBits);
  Value roundedUp =
      arith::CmpIOp::create(b, arith::CmpIPredicate::ugt, backBits, xBits);
  // Stepping a rounded-up `r` down (an Inf to max-finite) truncates it.
  Value bits = arith::BitcastOp::create(b, i64Ty, r);
  Value up = arith::ExtUIOp::create(b, i64Ty, roundedUp);
  Value truncated = arith::SubIOp::create(b, bits, up);
  Value sticky = arith::ExtUIOp::create(b, i64Ty, inexact);
  Value odd = arith::OrIOp::create(b, truncated, sticky);
  if (nnan)
    return odd;

  // A NaN keeps its sign and the top 52 of its 112 payload bits, quieted.
  Value c60 = createConst(loc, i128Ty, 60, rewriter);
  Value shifted = arith::ShRUIOp::create(b, xBits, c60);
  Value top = arith::TruncIOp::create(b, i64Ty, shifted);
  Value payloadMask =
      createAPIntConst(loc, i64Ty, APInt::getLowBitsSet(64, 52), rewriter);
  Value payload = arith::AndIOp::create(b, top, payloadMask);
  Value zero = createConst(loc, i128Ty, 0, rewriter);
  Value isNegative =
      arith::CmpIOp::create(b, arith::CmpIPredicate::slt, xBits, zero);
  Value negQNaN =
      createAPIntConst(loc, i64Ty, APInt(64, 0xFFF8000000000000), rewriter);
  Value posQNaN =
      createAPIntConst(loc, i64Ty, APInt(64, 0x7FF8000000000000), rewriter);
  Value qNaN = arith::SelectOp::create(b, isNegative, negQNaN, posQNaN);
  Value nanBits = arith::OrIOp::create(b, payload, qNaN);
  Value absMask =
      createAPIntConst(loc, i128Ty, APInt::getLowBitsSet(128, 127), rewriter);
  Value mag = arith::AndIOp::create(b, xBits, absMask);
  Value infBits =
      createAPIntConst(loc, i128Ty, APInt(128, 0x7FFF).shl(112), rewriter);
  Value isNaN =
      arith::CmpIOp::create(b, arith::CmpIPredicate::ugt, mag, infBits);
  return arith::SelectOp::create(b, isNaN, nanBits, odd);
}

/// Rounds `operand` once in `rnd` to `resultTy`'s narrow element type.
static Value emitNarrowFloatRounding(ImplicitLocOpBuilder &b,
                                     PatternRewriter &rewriter, Value operand,
                                     Type resultTy, arith::RoundingMode rnd,
                                     arith::FastMathFlagsAttr fastmath) {
  Type operandTy = operand.getType();
  auto srcETy = cast<FloatType>(getElementTypeOrSelf(operandTy));
  auto dstETy = cast<FloatType>(getElementTypeOrSelf(resultTy));
  FloatType wideETy = getTruncSourceType(srcETy, dstETy);
  const arith::FloatFormatSpec wide =
      arith::floatFormatSpecFromFloatType(wideETy);
  const arith::FloatFormatSpec dst =
      arith::floatFormatSpecFromFloatType(dstETy);
  Type carrierTy =
      cloneToShapedType(operandTy, getCarrierIntType(b, wide, dst));
  arith::FastMathFlagsAttr nnan = getNNaNOnly(b.getContext(), fastmath);

  Value srcBits;
  if (isa<Float8E8M0FNUType>(srcETy)) {
    // emitNarrowFloatExt needs a zero, which f8E8M0FNU lacks: decode its bits.
    Value code = arith::BitcastOp::create(
        b, cloneToShapedType(operandTy, b.getI8Type()), operand);
    srcBits = emitF8E8M0FNUToF32Bits(b, rewriter, code, /*withNaN=*/true);
  } else if (isNarrowFloat(srcETy)) {
    // A narrow source is decoded exactly: the truncation is the only rounding.
    const arith::FloatFormatSpec narrow =
        arith::floatFormatSpecFromFloatType(srcETy);
    Type narrowBitsTy =
        cloneToShapedType(operandTy, b.getIntegerType(narrow.sizeInBits));
    Value code = arith::BitcastOp::create(b, narrowBitsTy, operand);
    Value carried = arith::ExtUIOp::create(b, carrierTy, code);
    srcBits = arith::emitNarrowFloatExt(b, carried, narrow, wide);
  } else if (isa<Float80Type, Float128Type>(srcETy)) {
    srcBits = emitRoundToOddF64Bits(b, rewriter, operand, nnan);
  } else {
    Value wideValue = operand;
    if (srcETy != wideETy)
      wideValue = arith::ExtFOp::create(
          b, cloneToShapedType(operandTy, wideETy), operand, nnan);
    Type wideBitsTy =
        cloneToShapedType(operandTy, b.getIntegerType(wide.sizeInBits));
    srcBits = arith::BitcastOp::create(b, wideBitsTy, wideValue);
  }
  if (srcBits.getType() != carrierTy)
    srcBits = arith::ExtUIOp::create(b, carrierTy, srcBits);
  Value dstBits = arith::emitNarrowFloatTrunc(b, srcBits, wide, dst, rnd);
  Type dstBitsTy =
      cloneToShapedType(operandTy, b.getIntegerType(dst.sizeInBits));
  Value dstCode = arith::TruncIOp::create(b, dstBitsTy, dstBits);
  return arith::BitcastOp::create(b, resultTy, dstCode);
}

/// Extends the narrow float `operand` to `resultTy` via getExtResultType.
static Value emitNarrowFloatExtension(ImplicitLocOpBuilder &b, Value operand,
                                      Type resultTy,
                                      arith::FastMathFlagsAttr fastmath) {
  Type operandTy = operand.getType();
  auto srcETy = cast<FloatType>(getElementTypeOrSelf(operandTy));
  auto dstETy = cast<FloatType>(getElementTypeOrSelf(resultTy));
  FloatType wideETy = getExtResultType(srcETy, dstETy);
  const arith::FloatFormatSpec src =
      arith::floatFormatSpecFromFloatType(srcETy);
  const arith::FloatFormatSpec wide =
      arith::floatFormatSpecFromFloatType(wideETy);
  Type carrierTy =
      cloneToShapedType(operandTy, getCarrierIntType(b, src, wide));

  Type srcBitsTy =
      cloneToShapedType(operandTy, b.getIntegerType(src.sizeInBits));
  Value code = arith::BitcastOp::create(b, srcBitsTy, operand);
  Value carried = arith::ExtUIOp::create(b, carrierTy, code);
  Value bits = arith::emitNarrowFloatExt(b, carried, src, wide);
  Type wideBitsTy =
      cloneToShapedType(operandTy, b.getIntegerType(wide.sizeInBits));
  if (wideBitsTy != carrierTy)
    bits = arith::TruncIOp::create(b, wideBitsTy, bits);
  Value result =
      arith::BitcastOp::create(b, cloneToShapedType(resultTy, wideETy), bits);
  if (wideETy.getWidth() < dstETy.getWidth())
    result = arith::ExtFOp::create(b, resultTy, result, fastmath);
  else if (wideETy != dstETy)
    result = arith::TruncFOp::create(b, resultTy, result,
                                     /*roundingmode=*/nullptr, fastmath);
  return result;
}

namespace {

template <typename OpTy, arith::CmpIPredicate pred>
struct MaxMinIOpConverter : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    Value cmp = arith::CmpIOp::create(rewriter, op.getLoc(), pred, lhs, rhs);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, cmp, lhs, rhs);
    return success();
  }
};

template <typename OpTy, arith::CmpFPredicate pred>
struct MaximumMinimumFOpConverter : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    Location loc = op.getLoc();
    // If any operand is NaN, 'cmp' will be true (and 'select' returns 'lhs').
    static_assert(pred == arith::CmpFPredicate::UGT ||
                      pred == arith::CmpFPredicate::ULT,
                  "pred must be either UGT or ULT");
    Value cmp = arith::CmpFOp::create(rewriter, loc, pred, lhs, rhs);
    Value select = arith::SelectOp::create(rewriter, loc, cmp, lhs, rhs);

    // Handle the case where rhs is NaN: 'isNaN(rhs) ? rhs : select'.
    Value isNaN = arith::CmpFOp::create(rewriter, loc,
                                        arith::CmpFPredicate::UNO, rhs, rhs);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isNaN, rhs, select);
    return success();
  }
};

template <typename OpTy, arith::CmpFPredicate pred>
struct MaxNumMinNumFOpConverter : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    Location loc = op.getLoc();
    // If any operand is NaN, 'cmp' will be true (and 'select' returns 'lhs').
    static_assert(pred == arith::CmpFPredicate::UGT ||
                      pred == arith::CmpFPredicate::ULT,
                  "pred must be either UGT or ULT");
    Value cmp = arith::CmpFOp::create(rewriter, loc, pred, lhs, rhs);
    Value select = arith::SelectOp::create(rewriter, loc, cmp, lhs, rhs);

    // Handle the case where lhs is NaN: 'isNaN(lhs) ? rhs : select'.
    Value isNaN = arith::CmpFOp::create(rewriter, loc,
                                        arith::CmpFPredicate::UNO, lhs, lhs);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isNaN, rhs, select);
    return success();
  }
};

struct BFloat16ExtFOpConverter : public OpRewritePattern<arith::ExtFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const final {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto operand = op.getOperand();
    Type operandTy = operand.getType();
    Type resultTy = op.getType();
    Type operandETy = getElementTypeOrSelf(operandTy);
    Type resultETy = getElementTypeOrSelf(resultTy);

    if (!operandETy.isBF16() || !resultETy.isF32()) {
      return rewriter.notifyMatchFailure(op, "not a ext of bf16 to f32.");
    }

    Type i16Ty = cloneToShapedType(operandTy, b.getI16Type());
    Type i32Ty = cloneToShapedType(operandTy, b.getI32Type());

    Value bitcast = arith::BitcastOp::create(b, i16Ty, operand);
    Value exti = arith::ExtUIOp::create(b, i32Ty, bitcast);

    Value c16 = createConst(op.getLoc(), i32Ty, 16, rewriter);
    Value shl = arith::ShLIOp::create(b, exti, c16);
    Value result = arith::BitcastOp::create(b, resultTy, shl);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct BFloat16TruncFOpConverter : public OpRewritePattern<arith::TruncFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const final {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto operand = op.getOperand();
    Type operandTy = operand.getType();
    Type resultTy = op.getType();
    Type operandETy = getElementTypeOrSelf(operandTy);
    Type resultETy = getElementTypeOrSelf(resultTy);

    if (!operandETy.isF32() || !resultETy.isBF16()) {
      return rewriter.notifyMatchFailure(op, "not a trunc of f32 to bf16.");
    }

    if (op.getRoundingmodeAttr()) {
      return rewriter.notifyMatchFailure(
          op, "only applicable to default rounding mode.");
    }

    Type i16Ty = cloneToShapedType(operandTy, b.getI16Type());
    Type i32Ty = cloneToShapedType(operandTy, b.getI32Type());

    // Algorithm borrowed from this excellent code:
    // https://github.com/pytorch/pytorch/blob/e1502c0cdbfd17548c612f25d5a65b1e4b86224d/c10/util/BFloat16.h#L60-L79
    // There is a magic idea there, to let the addition of the rounding_bias to
    // the mantissa simply overflow into the exponent bits. It's a bit of an
    // aggressive, obfuscating optimization, but it is well-tested code, and it
    // results in more concise and efficient IR.
    // The case of NaN is handled separately (see isNaN and the final select).
    // The case of infinities is NOT handled separately, which deserves an
    // explanation. As the encoding of infinities has zero mantissa, the
    // rounding-bias addition never carries into the exponent so that just gets
    // truncated away, and as bfloat16 and float32 have the same number of
    // exponent bits, that simple truncation is the desired outcome for
    // infinities.
    Value isNan =
        arith::CmpFOp::create(b, arith::CmpFPredicate::UNE, operand, operand);
    // Constant used to make the rounding bias.
    Value c7FFF = createConst(op.getLoc(), i32Ty, 0x7fff, rewriter);
    // Constant used to generate a quiet NaN.
    Value c7FC0I16 = createConst(op.getLoc(), i16Ty, 0x7fc0, rewriter);
    // Small constants used to address bits.
    Value c16 = createConst(op.getLoc(), i32Ty, 16, rewriter);
    Value c1 = createConst(op.getLoc(), i32Ty, 1, rewriter);
    // Reinterpret the input f32 value as bits.
    Value bitcast = arith::BitcastOp::create(b, i32Ty, operand);
    // Read bit 16 as a value in {0,1}.
    Value bit16 =
        arith::AndIOp::create(b, arith::ShRUIOp::create(b, bitcast, c16), c1);
    // Determine the rounding bias to add as either 0x7fff or 0x8000 depending
    // on bit 16, implementing the tie-breaking "to nearest even".
    Value roundingBias = arith::AddIOp::create(b, bit16, c7FFF);
    // Add the rounding bias. Generally we want this to be added to the
    // mantissa, but nothing prevents this to from carrying into the exponent
    // bits, which would feel like a bug, but this is the magic trick here:
    // when that happens, the mantissa gets reset to zero and the exponent
    // gets incremented by the carry... which is actually exactly what we
    // want.
    Value biased = arith::AddIOp::create(b, bitcast, roundingBias);
    // Now that the rounding-bias has been added, truncating the low bits
    // yields the correctly rounded result.
    Value biasedAndShifted = arith::ShRUIOp::create(b, biased, c16);
    Value normalCaseResultI16 =
        arith::TruncIOp::create(b, i16Ty, biasedAndShifted);
    // Select either the above-computed result, or a quiet NaN constant
    // if the input was NaN.
    Value select =
        arith::SelectOp::create(b, isNan, c7FC0I16, normalCaseResultI16);
    Value result = arith::BitcastOp::create(b, resultTy, select);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct F8E8M0ExtFOpConverter : public OpRewritePattern<arith::ExtFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const final {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value operand = op.getOperand();
    Type operandTy = operand.getType();
    Type resultTy = op.getType();
    Type operandETy = getElementTypeOrSelf(operandTy);
    Type resultETy = getElementTypeOrSelf(resultTy);

    if (!llvm::isa<Float8E8M0FNUType>(operandETy)) {
      return rewriter.notifyMatchFailure(op, "not a ext of F8E8M0FNU");
    }

    Type i8Ty = cloneToShapedType(operandTy, b.getI8Type());
    Type i32Ty = cloneToShapedType(operandTy, b.getI32Type());
    Type f32Ty = cloneToShapedType(operandTy, b.getF32Type());

    Value bitcast = arith::BitcastOp::create(b, i8Ty, operand);
    // If FastMathFlag allows no NaN checks, skip it
    auto fastMath = op.getFastmathAttr();
    bool NoNaN = fastMath
                     ? (fastMath.getValue() & arith::FastMathFlags::nnan) ==
                           arith::FastMathFlags::nnan
                     : false;
    if (resultETy.isF64() || isa<Float80Type, Float128Type>(resultETy)) {
      // Every finite code c is the normal f64 2^(c - 127): (c + 896) << 52.
      Type i64Ty = cloneToShapedType(operandTy, b.getI64Type());
      Type f64Ty = cloneToShapedType(operandTy, b.getF64Type());
      Value code64 = arith::ExtUIOp::create(b, i64Ty, bitcast);
      Value cBias = createConst(op.getLoc(), i64Ty, 1023 - 127, rewriter);
      Value biasedExp = arith::AddIOp::create(b, code64, cBias);
      Value cF64MantissaWidth = createConst(op.getLoc(), i64Ty, 52, rewriter);
      Value f64Bits = arith::ShLIOp::create(b, biasedExp, cF64MantissaWidth);
      if (!NoNaN) {
        Value cF8NaN = createConst(op.getLoc(), i8Ty, 0xff, rewriter);
        Value cF64NaN = createAPIntConst(
            op.getLoc(), i64Ty, APInt(64, 0x7ff8000000000000), rewriter);
        Value isNan =
            arith::CmpIOp::create(b, arith::CmpIPredicate::eq, bitcast, cF8NaN);
        f64Bits = arith::SelectOp::create(b, isNan, cF64NaN, f64Bits);
      }
      Value result = arith::BitcastOp::create(b, f64Ty, f64Bits);
      if (resultETy.getIntOrFloatBitWidth() > 64)
        result =
            arith::ExtFOp::create(b, resultTy, result, op.getFastmathAttr());
      rewriter.replaceOp(op, result);
      return success();
    }

    Value f32Bits =
        emitF8E8M0FNUToF32Bits(b, rewriter, bitcast, /*withNaN=*/!NoNaN);
    if (isa<BFloat16Type>(resultETy)) {
      // bf16 has f32's exponent range: its bits are the f32 bits' high half.
      Type i16Ty = cloneToShapedType(operandTy, b.getI16Type());
      Value cHalf = createConst(op.getLoc(), i32Ty, 16, rewriter);
      Value high = arith::ShRUIOp::create(b, f32Bits, cHalf);
      Value bf16Bits = arith::TruncIOp::create(b, i16Ty, high);
      rewriter.replaceOp(op, arith::BitcastOp::create(b, resultTy, bf16Bits));
      return success();
    }
    Value result = arith::BitcastOp::create(b, f32Ty, f32Bits);
    if (resultETy.getIntOrFloatBitWidth() < 32) {
      result = arith::TruncFOp::create(b, resultTy, result, nullptr,
                                       op.getFastmathAttr());
    } else if (resultETy.getIntOrFloatBitWidth() > 32) {
      result = arith::ExtFOp::create(b, resultTy, result, op.getFastmathAttr());
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/*
TruncF to F8E8M0 is expected to extract exponent bits out of F32 type
Since All kinds of Infs and NaNs are mapped to same exponent bits in F32 type,
they all map to NaN in F8E8M0 Type.
*/
struct F8E8M0TruncFOpConverter : public OpRewritePattern<arith::TruncFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const final {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value operand = op.getOperand();
    Type operandTy = operand.getType();
    Type operandETy = getElementTypeOrSelf(operandTy);
    Type resultTy = op.getType();
    Type resultETy = getElementTypeOrSelf(resultTy);
    if (!llvm::isa<Float8E8M0FNUType>(resultETy)) {
      return rewriter.notifyMatchFailure(op, "not a truncf to f8E8M0FNU");
    }

    if (op.getRoundingmodeAttr()) {
      return rewriter.notifyMatchFailure(
          op, "only applicable to default rounding mode.");
    }

    // Taking the exponent rounds toward zero: do so from a wider source's bits.
    if (operandETy.getIntOrFloatBitWidth() > 32) {
      rewriter.replaceOp(
          op, emitNarrowFloatRounding(b, rewriter, operand, resultTy,
                                      arith::RoundingMode::toward_zero,
                                      op.getFastmathAttr()));
      return success();
    }

    Type i8Ty = cloneToShapedType(operandTy, b.getI8Type());
    Type i32Ty = cloneToShapedType(operandTy, b.getI32Type());
    Type f32Ty = cloneToShapedType(operandTy, b.getF32Type());

    if (operandETy.getIntOrFloatBitWidth() < 32) {
      operand = arith::ExtFOp::create(
          b, f32Ty, operand, getNNaNOnly(b.getContext(), op.getFastmathAttr()));
    }
    Value f32Bits = arith::BitcastOp::create(b, i32Ty, operand);
    Value cF32MantissaWidth = createConst(op->getLoc(), i32Ty, 23, rewriter);
    Value f32SignExp = arith::ShRUIOp::create(b, f32Bits, cF32MantissaWidth);
    Value exp8Bits = arith::TruncIOp::create(b, i8Ty, f32SignExp);
    Value result = arith::BitcastOp::create(b, resultTy, exp8Bits);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Expands a truncf into a narrow float (see canExpandNarrowFloat).
struct NarrowFloatTruncFOpConverter : public OpRewritePattern<arith::TruncFOp> {
  NarrowFloatTruncFOpConverter(MLIRContext *ctx, ArrayRef<TypeID> narrowTypes)
      : OpRewritePattern(ctx), narrowTypes(narrowTypes) {}

  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const final {
    Value operand = op.getOperand();
    Type resultTy = op.getType();
    if (!canExpandNarrowFloat(
            /*isExt=*/false, getElementTypeOrSelf(operand.getType()),
            getElementTypeOrSelf(resultTy), resultTy, narrowTypes))
      return rewriter.notifyMatchFailure(op, "not a narrow-float truncf");
    // Without a mode, a truncf into a format without a zero takes the exponent
    // (populateExpandF8E8M0Patterns), not the default rounding.
    auto resultETy = cast<FloatType>(getElementTypeOrSelf(resultTy));
    if (!op.getRoundingmode() && !resultETy.getFloatSemantics().hasZero)
      return rewriter.notifyMatchFailure(op, "no rounding mode");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    arith::RoundingMode rnd =
        op.getRoundingmode().value_or(arith::RoundingMode::to_nearest_even);
    Value result = emitNarrowFloatRounding(b, rewriter, operand, resultTy, rnd,
                                           op.getFastmathAttr());
    rewriter.replaceOp(op, result);
    return success();
  }

  SmallVector<TypeID> narrowTypes;
};

/// Expands an extf from a narrow float (see canExpandNarrowFloat).
struct NarrowFloatExtFOpConverter : public OpRewritePattern<arith::ExtFOp> {
  NarrowFloatExtFOpConverter(MLIRContext *ctx, ArrayRef<TypeID> narrowTypes)
      : OpRewritePattern(ctx), narrowTypes(narrowTypes) {}

  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const final {
    Value operand = op.getOperand();
    Type operandETy = getElementTypeOrSelf(operand.getType());
    Type resultTy = op.getType();
    Type resultETy = getElementTypeOrSelf(resultTy);
    if (!canExpandNarrowFloat(/*isExt=*/true, operandETy, resultETy, resultTy,
                              narrowTypes))
      return rewriter.notifyMatchFailure(op, "not a narrow-float extf");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value result =
        isExactNarrowPair(cast<FloatType>(operandETy),
                          cast<FloatType>(resultETy))
            ? emitNarrowFloatRounding(b, rewriter, operand, resultTy,
                                      arith::RoundingMode::to_nearest_even,
                                      op.getFastmathAttr())
            : emitNarrowFloatExtension(b, operand, resultTy,
                                       op.getFastmathAttr());
    rewriter.replaceOp(op, result);
    return success();
  }

  SmallVector<TypeID> narrowTypes;
};

struct ScalingExtFOpConverter : public OpRewritePattern<arith::ScalingExtFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(arith::ScalingExtFOp op,
                                PatternRewriter &rewriter) const final {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value inputOperand = op.getIn();
    Value scaleOperand = op.getScale();
    Type scaleTy = scaleOperand.getType();
    Type scaleETy = getElementTypeOrSelf(scaleOperand);
    // allow implicit exponent extraction from 16/32 bits floats
    if (scaleETy.getIntOrFloatBitWidth() >= 16) {
      scaleETy = b.getF8E8M0Type();
      scaleTy = cloneToShapedType(scaleTy, scaleETy);
      scaleOperand = arith::TruncFOp::create(b, scaleTy, scaleOperand, nullptr,
                                             op.getFastmathAttr());
    }
    // Catch scale types like f8E5M2.
    if (!llvm::isa<Float8E8M0FNUType>(scaleETy)) {
      return rewriter.notifyMatchFailure(
          op, "scaling_extf is using scales of type which can not be converted "
              "to f8E8M0FNU");
    }
    Type resultTy = op.getType();
    // extf on scale will essentially create floating point number
    // of type resulTy that is 2^scale and will also propagate NaNs
    Value scaleExt =
        arith::ExtFOp::create(b, resultTy, scaleOperand, op.getFastmathAttr());
    Value inputExt =
        arith::ExtFOp::create(b, resultTy, inputOperand, op.getFastmathAttr());
    Value result =
        arith::MulFOp::create(b, inputExt, scaleExt, op.getFastmathAttr());
    rewriter.replaceOp(op, result);
    return success();
  }
};

/*
Expands arith.ScalingTruncFOp(in, scale) into
  scale = arith.truncf(scale) : scaleTy -> f8E8M0FNU
  result = arith.truncf(in / (2^scale))
 */
struct ScalingTruncFOpConverter
    : public OpRewritePattern<arith::ScalingTruncFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(arith::ScalingTruncFOp op,
                                PatternRewriter &rewriter) const final {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value inputOperand = op.getIn();
    Value scaleOperand = op.getScale();
    Type scaleTy = scaleOperand.getType();
    Type scaleETy = getElementTypeOrSelf(scaleOperand);
    // allow implicit exponent extraction from 16/32 bits floats
    if (scaleETy.getIntOrFloatBitWidth() >= 16) {
      scaleETy = b.getF8E8M0Type();
      scaleTy = cloneToShapedType(scaleTy, scaleETy);
      scaleOperand = arith::TruncFOp::create(b, scaleTy, scaleOperand, nullptr,
                                             op.getFastmathAttr());
    }
    if (!llvm::isa<Float8E8M0FNUType>(scaleETy)) {
      return rewriter.notifyMatchFailure(
          op, "scaling_truncf is using scales type which can not be converted "
              "to f8E8M0FNU");
    }
    Type resultTy = op.getType();
    Type inputTy = inputOperand.getType();
    // this will create a floating point number of type
    // inputTy that is 2^scale and will also propagate NaNs
    scaleOperand =
        arith::ExtFOp::create(b, inputTy, scaleOperand, op.getFastmathAttr());
    Value result = arith::DivFOp::create(b, inputOperand, scaleOperand,
                                         op.getFastmathAttr());
    Value resultCast = arith::TruncFOp::create(
        b, resultTy, result, op.getRoundingmodeAttr(), op.getFastmathAttr());
    rewriter.replaceOp(op, resultCast);
    return success();
  }
};

/// Expands `arith.flush_denormals` into integer arithmetic.
///
/// For an IEEE-like floating-point value with a sign|exponent|mantissa bit
/// layout, a value is denormal iff its biased exponent field is zero and its
/// stored mantissa is non-zero. When the exponent field is zero, the value is
/// either pos/neg 0 (mantissa = 0) or a denormal (mantissa != 0); in both
/// cases, clearing the mantissa bits produces the desired sign-preserved zero
/// (a no-op for pos/neg 0, a flush for denormals). When the exponent field is
/// non-zero, the value passes through unchanged.
///
/// Pseudocode:
///   bits        = bitcast(x, iN)
///   expIsZero   = (bits & expMask) == 0
///   cleared     = bits & ~manMask
///   resultBits  = select(expIsZero, cleared, bits)
///   result      = bitcast(resultBits, floatTy)
struct FlushDenormalsOpConverter
    : public OpRewritePattern<arith::FlushDenormalsOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(arith::FlushDenormalsOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    Value operand = op.getOperand();
    Type operandTy = operand.getType();
    auto floatTy = dyn_cast<FloatType>(getElementTypeOrSelf(operandTy));
    if (!floatTy)
      return rewriter.notifyMatchFailure(op, "operand is not a float type");

    const llvm::fltSemantics &sem = floatTy.getFloatSemantics();
    // Restrict to IEEE-like encodings, where the sign bit is the MSB and
    // denormals are exactly "biased exponent == 0 and non-zero mantissa".
    if (!llvm::APFloatBase::isIEEELikeFP(sem))
      return rewriter.notifyMatchFailure(
          op, "only IEEE-like floating-point types are supported");

    unsigned totalBits = llvm::APFloatBase::semanticsSizeInBits(sem);
    unsigned precision = llvm::APFloatBase::semanticsPrecision(sem);
    // Stored mantissa bits = precision - 1 (implicit leading bit not stored).
    // Exponent field bits = totalBits - 1 (sign) - storedMantissa.
    if (precision < 1 || precision > totalBits)
      return rewriter.notifyMatchFailure(op, "unexpected float semantics");
    unsigned mantissaBits = precision - 1;
    unsigned expBits = totalBits - 1 - mantissaBits;
    if (expBits == 0 || mantissaBits == 0)
      return rewriter.notifyMatchFailure(
          op, "degenerate float encoding has no exponent or mantissa");

    Type intTy =
        cloneToShapedType(operandTy, rewriter.getIntegerType(totalBits));
    Value bits = arith::BitcastOp::create(b, intTy, operand);
    APInt expMaskVal =
        APInt::getBitsSet(totalBits, mantissaBits, mantissaBits + expBits);
    APInt clearMantissaMaskVal = ~APInt::getLowBitsSet(totalBits, mantissaBits);
    APInt zeroVal = APInt::getZero(totalBits);
    Value expMask = createAPIntConst(loc, intTy, expMaskVal, rewriter);
    Value clearMantissaMask =
        createAPIntConst(loc, intTy, clearMantissaMaskVal, rewriter);
    Value zero = createAPIntConst(loc, intTy, zeroVal, rewriter);

    // expField == 0
    Value expField = arith::AndIOp::create(b, bits, expMask);
    Value expIsZero =
        arith::CmpIOp::create(b, arith::CmpIPredicate::eq, expField, zero);

    // Clear mantissa bits: when exp == 0, this produces pos/neg 0.0.
    Value cleared = arith::AndIOp::create(b, bits, clearMantissaMask);
    Value resultBits = arith::SelectOp::create(b, expIsZero, cleared, bits);
    Value result = arith::BitcastOp::create(b, operandTy, resultBits);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithExpandOpsPass
    : public arith::impl::ArithExpandOpsPassBase<ArithExpandOpsPass> {
  using ArithExpandOpsPassBase::ArithExpandOpsPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    arith::populateExpandScalingExtTruncPatterns(patterns);

    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<vector::VectorDialect>();

    // clang-format off
    target.addIllegalOp<
      arith::CeilDivSIOp,
      arith::CeilDivUIOp,
      arith::FloorDivSIOp,
      arith::ScalingExtFOp,
      arith::ScalingTruncFOp
    >();
    // clang-format on

    // The min/max ops also have a direct arith-to-llvm lowering to the
    // `llvm.intr.maximum`/`minimum`/... (and smax/umax/...) intrinsics, which
    // are a single hardware instruction on many targets. Only expand them into
    // cmpf/cmpi + select when requested, so pipelines that run arith-to-llvm
    // can keep the intrinsic lowering. The float and integer ops are gated
    // separately.
    if (includeMinMaxF) {
      arith::populateExpandMinMaxFPatterns(patterns);
      // clang-format off
      target.addIllegalOp<
        arith::MaximumFOp,
        arith::MinimumFOp,
        arith::MaxNumFOp,
        arith::MinNumFOp
      >();
      // clang-format on
    }
    if (includeMinMaxI) {
      arith::populateExpandMinMaxIPatterns(patterns);
      // clang-format off
      target.addIllegalOp<
        arith::MaxSIOp,
        arith::MaxUIOp,
        arith::MinSIOp,
        arith::MinUIOp
      >();
      // clang-format on
    }

    if (includeBf16)
      arith::populateExpandBFloat16Patterns(patterns);
    if (includeF8E8M0)
      arith::populateExpandF8E8M0Patterns(patterns);
    const std::pair<bool, TypeID> narrowFloatOptions[] = {
        {includeF4E2M1, TypeID::get<Float4E2M1FNType>()},
        {includeF8E5M2, TypeID::get<Float8E5M2Type>()},
        {includeF8E4M3FN, TypeID::get<Float8E4M3FNType>()},
        {includeF8E4M3, TypeID::get<Float8E4M3Type>()},
        {includeF8E4M3FNUZ, TypeID::get<Float8E4M3FNUZType>()},
        {includeF8E4M3B11FNUZ, TypeID::get<Float8E4M3B11FNUZType>()},
        {includeF8E5M2FNUZ, TypeID::get<Float8E5M2FNUZType>()},
        {includeF8E3M4, TypeID::get<Float8E3M4Type>()},
        {includeF6E2M3FN, TypeID::get<Float6E2M3FNType>()},
        {includeF6E3M2FN, TypeID::get<Float6E3M2FNType>()}};
    SmallVector<TypeID> narrowTypes;
    for (auto [enabled, typeID] : narrowFloatOptions)
      if (enabled)
        narrowTypes.push_back(typeID);
    arith::populateExpandNarrowFloatPatterns(patterns, narrowTypes);
    if (includeFlushDenormals) {
      arith::populateExpandFlushDenormalsPatterns(patterns);
      // Only IEEE-like floating-point types are expanded by the pattern;
      // leave `arith.flush_denormals` on other types alone.
      target.addDynamicallyLegalOp<arith::FlushDenormalsOp>(
          [](arith::FlushDenormalsOp op) {
            auto floatTy =
                dyn_cast<FloatType>(getElementTypeOrSelf(op.getType()));
            if (!floatTy)
              return true;
            return !llvm::APFloatBase::isIEEELikeFP(
                floatTy.getFloatSemantics());
          });
    }

    target.addDynamicallyLegalOp<arith::ExtFOp>([=](arith::ExtFOp op) {
      Type inETy = getElementTypeOrSelf(op.getOperand().getType());
      Type outETy = getElementTypeOrSelf(op.getType());
      bool legalTypes = true;
      if (includeBf16)
        legalTypes &= !(inETy.isBF16() && outETy.isF32());
      if (includeF8E8M0)
        legalTypes &= !llvm::isa<Float8E8M0FNUType>(inETy);
      legalTypes &= !canExpandNarrowFloat(/*isExt=*/true, inETy, outETy,
                                          op.getType(), narrowTypes);
      return legalTypes;
    });

    target.addDynamicallyLegalOp<arith::TruncFOp>([=](arith::TruncFOp op) {
      Type inETy = getElementTypeOrSelf(op.getOperand().getType());
      Type outETy = getElementTypeOrSelf(op.getType());
      bool legalTypes = true;
      if (includeBf16)
        legalTypes &= !(inETy.isF32() && outETy.isBF16());
      if (includeF8E8M0)
        legalTypes &= !(llvm::isa<Float8E8M0FNUType>(outETy));
      legalTypes &= !canExpandNarrowFloat(/*isExt=*/false, inETy, outETy,
                                          op.getType(), narrowTypes);
      return legalTypes;
    });

    // clang-format on
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::arith::populateExpandBFloat16Patterns(RewritePatternSet &patterns) {
  patterns.add<BFloat16ExtFOpConverter, BFloat16TruncFOpConverter>(
      patterns.getContext());
}

void mlir::arith::populateExpandNarrowFloatPatterns(
    RewritePatternSet &patterns, ArrayRef<TypeID> narrowTypes) {
  if (narrowTypes.empty())
    return;
  patterns.add<NarrowFloatTruncFOpConverter, NarrowFloatExtFOpConverter>(
      patterns.getContext(), narrowTypes);
}

void mlir::arith::populateExpandF4E2M1Patterns(RewritePatternSet &patterns) {
  populateExpandNarrowFloatPatterns(patterns, TypeID::get<Float4E2M1FNType>());
}

void mlir::arith::populateExpandF8E5M2Patterns(RewritePatternSet &patterns) {
  populateExpandNarrowFloatPatterns(patterns, TypeID::get<Float8E5M2Type>());
}

void mlir::arith::populateExpandF8E4M3FNPatterns(RewritePatternSet &patterns) {
  populateExpandNarrowFloatPatterns(patterns, TypeID::get<Float8E4M3FNType>());
}

void mlir::arith::populateExpandF8E8M0Patterns(RewritePatternSet &patterns) {
  patterns.add<F8E8M0ExtFOpConverter, F8E8M0TruncFOpConverter>(
      patterns.getContext());
  populateExpandNarrowFloatPatterns(patterns, TypeID::get<Float8E8M0FNUType>());
}

void mlir::arith::populateExpandScalingExtTruncPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ScalingExtFOpConverter, ScalingTruncFOpConverter>(
      patterns.getContext());
}

void mlir::arith::populateExpandFlushDenormalsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FlushDenormalsOpConverter>(patterns.getContext());
}

void mlir::arith::populateExpandMinMaxFPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    MaximumMinimumFOpConverter<MaximumFOp, arith::CmpFPredicate::UGT>,
    MaximumMinimumFOpConverter<MinimumFOp, arith::CmpFPredicate::ULT>,
    MaxNumMinNumFOpConverter<MaxNumFOp, arith::CmpFPredicate::UGT>,
    MaxNumMinNumFOpConverter<MinNumFOp, arith::CmpFPredicate::ULT>
   >(patterns.getContext());
  // clang-format on
}

void mlir::arith::populateExpandMinMaxIPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    MaxMinIOpConverter<MaxSIOp, arith::CmpIPredicate::sgt>,
    MaxMinIOpConverter<MaxUIOp, arith::CmpIPredicate::ugt>,
    MaxMinIOpConverter<MinSIOp, arith::CmpIPredicate::slt>,
    MaxMinIOpConverter<MinUIOp, arith::CmpIPredicate::ult>
   >(patterns.getContext());
  // clang-format on
}

void mlir::arith::populateExpandMinMaxPatterns(RewritePatternSet &patterns) {
  populateExpandMinMaxFPatterns(patterns);
  populateExpandMinMaxIPatterns(patterns);
}

void mlir::arith::populateArithExpandOpsPatterns(RewritePatternSet &patterns) {
  populateCeilFloorDivExpandOpsPatterns(patterns);
  populateExpandScalingExtTruncPatterns(patterns);
  populateExpandMinMaxPatterns(patterns);
}
