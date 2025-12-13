//===- ExpandOps.cpp - Pass to legalize Arith ops for LLVM lowering --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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

/// Create a float constant.
static Value createFloatConst(Location loc, Type type, const APFloat &value,
                              PatternRewriter &rewriter) {
  auto attr = rewriter.getFloatAttr(getElementTypeOrSelf(type), value);
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

namespace {

/// Expands CeilDivUIOp (n, m) into
///  n == 0 ? 0 : ((n-1) / m) + 1
struct CeilDivUIOpConverter : public OpRewritePattern<arith::CeilDivUIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(arith::CeilDivUIOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value a = op.getLhs();
    Value b = op.getRhs();
    Value zero = createConst(loc, a.getType(), 0, rewriter);
    Value compare =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, a, zero);
    Value one = createConst(loc, a.getType(), 1, rewriter);
    Value minusOne = arith::SubIOp::create(rewriter, loc, a, one);
    Value quotient = arith::DivUIOp::create(rewriter, loc, minusOne, b);
    Value plusOne = arith::AddIOp::create(rewriter, loc, quotient, one);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, compare, zero, plusOne);
    return success();
  }
};

/// Expands CeilDivSIOp (a, b) into
/// z = a / b
/// if (z * b != a && (a < 0) == (b < 0)) {
///   return z + 1;
/// } else {
///   return z;
/// }
struct CeilDivSIOpConverter : public OpRewritePattern<arith::CeilDivSIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(arith::CeilDivSIOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type type = op.getType();
    Value a = op.getLhs();
    Value b = op.getRhs();

    Value zero = createConst(loc, type, 0, rewriter);
    Value one = createConst(loc, type, 1, rewriter);

    Value quotient = arith::DivSIOp::create(rewriter, loc, a, b);
    Value product = arith::MulIOp::create(rewriter, loc, quotient, b);
    Value notEqualDivisor = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ne, a, product);

    Value aNeg = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                       a, zero);
    Value bNeg = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                       b, zero);

    Value signEqual = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, aNeg, bNeg);
    Value cond =
        arith::AndIOp::create(rewriter, loc, notEqualDivisor, signEqual);

    Value quotientPlusOne = arith::AddIOp::create(rewriter, loc, quotient, one);

    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, cond, quotientPlusOne,
                                                 quotient);
    return success();
  }
};

/// Expands FloorDivSIOp (x, y) into
/// z = x / y
/// if (z * y != x && (x < 0) != (y < 0)) {
///   return  z - 1;
/// } else {
///   return z;
/// }
struct FloorDivSIOpConverter : public OpRewritePattern<arith::FloorDivSIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(arith::FloorDivSIOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type type = op.getType();
    Value a = op.getLhs();
    Value b = op.getRhs();

    Value quotient = arith::DivSIOp::create(rewriter, loc, a, b);
    Value product = arith::MulIOp::create(rewriter, loc, quotient, b);
    Value notEqualDivisor = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ne, a, product);
    Value zero = createConst(loc, type, 0, rewriter);

    Value aNeg = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                       a, zero);
    Value bNeg = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                       b, zero);

    Value signOpposite = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ne, aNeg, bNeg);
    Value cond =
        arith::AndIOp::create(rewriter, loc, notEqualDivisor, signOpposite);

    Value minusOne = createConst(loc, type, -1, rewriter);
    Value quotientMinusOne =
        arith::AddIOp::create(rewriter, loc, quotient, minusOne);

    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, cond, quotientMinusOne,
                                                 quotient);
    return success();
  }
};

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

/// In this implementation of extf we take advantage of some key patterns we
/// notice between the binary representation of an F4E2M1 value and its
/// corresponding value in F32.
///
/// Note: x is sign bit
/// | Binary | F4E2M1 | f32[23:32]
/// | x000   | 0.0    | x000 0000 00
/// | x001   | 0.5    | x011 1111 00
/// | x010   | 1.0    | x011 1111 10
/// | x011   | 1.5    | x011 1111 11
/// | x100   | 2.0    | x010 0000 00
/// | x101   | 3.0    | x010 0000 01
/// | x110   | 4.0    | x010 0000 10
/// | x111   | 6.0    | x010 0000 11
///
/// 1) There are only two versions of bits [25:31] in the f32 result
///     F4E2M1 bits[2:3] decide whether:
///       - F32 bits[25:31] = 0011 1111
///       - F32 bits[25:31] = 0010 0000
///     Exception is zero where
///       - F32 bits[25:31] = 0000 0000
///
/// 2) F4E2M1 bits[1:2] = F32 bits[23:24]
///     Exception is 0.5 where
///       - F4E2M1 bits[1:2] = 01, F32 bits[23:24] = 00
///
/// 3) F4E2M1 bits[4] = F32 bits[32] (sign bits are equal)
///
/// 4) F32 bits[1:22] = 0
struct F4E2M1ExtFOpConverter : public OpRewritePattern<arith::ExtFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    Value operand = op.getOperand();
    Type operandTy = operand.getType();
    Type resultTy = op.getType();
    Type operandETy = getElementTypeOrSelf(operandTy);
    Type resultETy = getElementTypeOrSelf(resultTy);

    if (!isa<Float4E2M1FNType>(operandETy))
      return rewriter.notifyMatchFailure(op, "not a ext of F4E2M1FN");

    Type f32Ty = cloneToShapedType(operandTy, b.getF32Type());
    Type i4Ty = cloneToShapedType(operandTy, b.getI4Type());
    Type i32Ty = cloneToShapedType(operandTy, b.getI32Type());
    Value i4Bits = arith::BitcastOp::create(b, i4Ty, operand);

    Value c0x0 = createConst(loc, i4Ty, 0x0, rewriter);
    Value c0x1 = createConst(loc, i4Ty, 0x1, rewriter);
    Value c0x2 = createConst(loc, i4Ty, 0x2, rewriter);
    Value c0x4 = createConst(loc, i4Ty, 0x4, rewriter);
    Value c0x7 = createConst(loc, i4Ty, 0x7, rewriter);

    Value i4BitsNoSign = arith::AndIOp::create(b, i4Bits, c0x7);

    // Set last Exponent bit and Mantissa.
    Value c0x00000014 = createConst(loc, i32Ty, 0x14, rewriter);
    Value bits1To24 = arith::ShLIOp::create(b, i4BitsNoSign, c0x2);
    Value isHalf =
        arith::CmpIOp::create(b, arith::CmpIPredicate::eq, i4BitsNoSign, c0x1);
    bits1To24 = arith::SelectOp::create(b, isHalf, c0x0, bits1To24);
    bits1To24 = arith::ExtUIOp::create(b, i32Ty, bits1To24);
    bits1To24 = arith::ShLIOp::create(b, bits1To24, c0x00000014);

    // Set first 7 bits of Exponent.
    Value zeroExpBits = createConst(loc, i32Ty, 0x00000000, rewriter);
    Value highExpBits = createConst(loc, i32Ty, 0x40000000, rewriter);
    Value lowExpBits = createConst(loc, i32Ty, 0x3f000000, rewriter);
    Value useLargerExp =
        arith::CmpIOp::create(b, arith::CmpIPredicate::uge, i4BitsNoSign, c0x4);
    Value bits25To31 =
        arith::SelectOp::create(b, useLargerExp, highExpBits, lowExpBits);
    Value zeroExp =
        arith::CmpIOp::create(b, arith::CmpIPredicate::eq, i4BitsNoSign, c0x0);
    bits25To31 = arith::SelectOp::create(b, zeroExp, zeroExpBits, bits25To31);

    // Set sign.
    Value c0x80000000 = createConst(loc, i32Ty, 0x80000000, rewriter);
    Value c0x8 = createConst(loc, i4Ty, 0x8, rewriter);
    Value negative =
        arith::CmpIOp::create(b, arith::CmpIPredicate::uge, i4Bits, c0x8);
    Value bit32 =
        arith::SelectOp::create(b, negative, c0x80000000, zeroExpBits);

    // Add segments together.
    Value bits1To31 = arith::AddIOp::create(b, bits1To24, bits25To31);
    Value bits1To32 = arith::AddIOp::create(b, bits1To31, bit32);
    Value result = arith::BitcastOp::create(b, f32Ty, bits1To32);
    if (!isa<Float32Type>(resultETy))
      result = arith::TruncFOp::create(b, resultTy, result);

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
    // create constants for NaNs
    Value cF8NaN = createConst(op.getLoc(), i8Ty, 0xff, rewriter);
    Value cF32NaN = createConst(op.getLoc(), i32Ty, 0xffffffff, rewriter);
    Value cF32MantissaWidth = createConst(op->getLoc(), i32Ty, 23, rewriter);

    Value exti = arith::ExtUIOp::create(b, i32Ty, bitcast);
    Value f32Bits = arith::ShLIOp::create(b, exti, cF32MantissaWidth);

    Value isNan =
        arith::CmpIOp::create(b, arith::CmpIPredicate::eq, bitcast, cF8NaN);
    // select for NaNs
    f32Bits = arith::SelectOp::create(b, isNan, cF32NaN, f32Bits);
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

/// Conversion from F32 to F4E2M1 according to the OCP Spec:
/// www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
///
/// The spec requiers us to perform Round to Nearest, Ties to Even.
///
/// This means that after rounding, we should break ties by choosing the option
/// which results in a mantissa of 0 in the least significant digit.
///
/// Table of representable values in F4E2M1:
///
/// Note: x is sign bit
/// | Binary | F4E2M1 | F32[23:32]
/// | x000   | 0.0    | x000 0000 00
/// | x001   | 0.5    | x011 1111 00
/// | x010   | 1.0    | x011 1111 10
/// | x011   | 1.5    | x011 1111 11
/// | x100   | 2.0    | x010 0000 00
/// | x101   | 3.0    | x010 0000 01
/// | x110   | 4.0    | x010 0000 10
/// | x111   | 6.0    | x010 0000 11
///
/// Conversion procedure:
///   Step 1: Clamp to representable bounds.
///   Step 2: Convert exponent by adjusting bias.
///   Step 3: Set mantissa to first bit.
///   Step 4: Special consideration for subnormal and zero exponent.
///   Step 5: Round up if necessary, if mantissa[1:] greater than 1000000 or
///   subnormal.
struct F4E2M1TruncFOpConverter : public OpRewritePattern<arith::TruncFOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    Value operand = op.getOperand();
    Type operandTy = operand.getType();
    Type resultTy = op.getType();
    Type operandETy = getElementTypeOrSelf(operandTy);
    Type resultETy = getElementTypeOrSelf(resultTy);

    Type i4Ty = cloneToShapedType(operandTy, b.getI4Type());
    Type i8Ty = cloneToShapedType(operandTy, b.getI8Type());
    Type i32Ty = cloneToShapedType(operandTy, b.getI32Type());
    Type f32Ty = cloneToShapedType(operandTy, b.getF32Type());

    if (!isa<Float4E2M1FNType>(resultETy))
      return rewriter.notifyMatchFailure(op, "not a trunc of F4E2M1FN");
    if (!isa<Float32Type>(operandETy))
      operand = arith::ExtFOp::create(b, f32Ty, operand);

    Value c0x1 = createConst(loc, i4Ty, 1, rewriter);
    Value c0x3 = createConst(loc, i4Ty, 3, rewriter);
    Value c0x00000016 = createConst(loc, i32Ty, 22, rewriter);
    Value c0x00 = createConst(loc, i8Ty, 0x00, rewriter);
    Value c0xff = createConst(loc, i8Ty, 0xff, rewriter);
    Value zeroExpBits = createConst(loc, i32Ty, 0, rewriter);

    // Step 0: Clamp to bounds.
    Value cHigherBound = createFloatConst(loc, f32Ty, APFloat(6.0f), rewriter);
    Value cLowerBound = createFloatConst(loc, f32Ty, APFloat(-6.0f), rewriter);
    Value operandClamped = arith::MinNumFOp::create(b, cHigherBound, operand);
    operandClamped = arith::MaxNumFOp::create(b, cLowerBound, operandClamped);
    Value f32Bits = arith::BitcastOp::create(b, i32Ty, operandClamped);

    // Step 1: Set sign bit.
    Value cF32ExpManWidth = createConst(loc, i32Ty, 31, rewriter); // 23
    Value f32Sign = arith::ShRUIOp::create(b, f32Bits, cF32ExpManWidth);
    Value f4Sign = arith::TruncIOp::create(b, i4Ty, f32Sign);
    Value f4Bits = arith::ShLIOp::create(b, f4Sign, c0x3);

    // Step 2: Convert exponent by adjusting bias.
    Value biasAdjustment = createConst(loc, i32Ty, 0x7e, rewriter);
    Value cF4MantissaWidth = c0x1;                                   // 1
    Value cF32MantissaWidth = createConst(loc, i32Ty, 23, rewriter); // 23
    Value f32SignExp = arith::ShRUIOp::create(b, f32Bits, cF32MantissaWidth);
    Value biasAdjustedSignExp =
        arith::SubIOp::create(b, f32SignExp, biasAdjustment);
    Value f4Exp = arith::TruncIOp::create(b, i4Ty, biasAdjustedSignExp);
    f4Exp = arith::ShLIOp::create(b, f4Exp, cF4MantissaWidth);
    f4Bits = arith::AddIOp::create(b, f4Bits, f4Exp);

    // Step 3: Set mantissa to first bit.
    Value cF32FirstBitMask = createConst(loc, i32Ty, 0x400000, rewriter);
    Value man1Bit = arith::AndIOp::create(b, f32Bits, cF32FirstBitMask);
    man1Bit = arith::ShRUIOp::create(b, man1Bit, c0x00000016);
    Value f4Man = arith::TruncIOp::create(b, i4Ty, man1Bit);
    f4Bits = arith::AddIOp::create(b, f4Bits, f4Man);

    // Step 4: Special consideration for conversion to 0.5.
    Value cF32MantissaMask = createConst(loc, i32Ty, 0x7fffff, rewriter);
    Value f8Exp = arith::TruncIOp::create(b, i8Ty, biasAdjustedSignExp);
    Value isSubnormal =
        arith::CmpIOp::create(b, arith::CmpIPredicate::sle, f8Exp, c0x00);
    Value isNegOneExp =
        arith::CmpIOp::create(b, arith::CmpIPredicate::eq, f8Exp, c0xff);
    Value man23Bits = arith::AndIOp::create(b, f32Bits, cF32MantissaMask);
    Value isNonZeroMan = arith::CmpIOp::create(b, arith::CmpIPredicate::ugt,
                                               man23Bits, zeroExpBits);
    Value roundToHalf = arith::AndIOp::create(b, isNegOneExp, isNonZeroMan);
    Value isZeroExp =
        arith::CmpIOp::create(b, arith::CmpIPredicate::eq, f8Exp, c0x00);
    Value subnormalF4Bits = createConst(loc, i4Ty, 0xf, rewriter);
    Value halfF4Bits = createConst(loc, i4Ty, 0x0, rewriter);
    Value subResult =
        arith::SelectOp::create(b, isSubnormal, subnormalF4Bits, f4Bits);
    subResult = arith::SelectOp::create(b, roundToHalf, halfF4Bits, subResult);
    f4Bits = arith::SelectOp::create(b, isZeroExp, f4Bits, subResult);

    // Step 5: Round up if necessary.
    Value cF32Last22BitMask = createConst(loc, i32Ty, 0x3fffff, rewriter);
    Value cRound = createConst(loc, i32Ty, 0x200000, rewriter); // 010 0000...
    Value man22Bits = arith::AndIOp::create(b, f32Bits, cF32Last22BitMask);
    Value shouldRound =
        arith::CmpIOp::create(b, arith::CmpIPredicate::uge, man22Bits, cRound);
    shouldRound = arith::OrIOp::create(b, shouldRound, isSubnormal);
    Value roundedF4Bits = arith::AddIOp::create(b, f4Bits, c0x1);
    f4Bits = arith::SelectOp::create(b, shouldRound, roundedF4Bits, f4Bits);

    Value result = arith::BitcastOp::create(b, resultTy, f4Bits);
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

    Type i8Ty = cloneToShapedType(operandTy, b.getI8Type());
    Type i32Ty = cloneToShapedType(operandTy, b.getI32Type());
    Type f32Ty = cloneToShapedType(operandTy, b.getF32Type());

    if (operandETy.getIntOrFloatBitWidth() < 32) {
      operand = arith::ExtFOp::create(b, f32Ty, operand, op.getFastmathAttr());
    } else if (operandETy.getIntOrFloatBitWidth() > 32) {
      operand = arith::TruncFOp::create(
          b, f32Ty, operand, op.getRoundingmodeAttr(), op.getFastmathAttr());
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

struct ArithExpandOpsPass
    : public arith::impl::ArithExpandOpsPassBase<ArithExpandOpsPass> {
  using ArithExpandOpsPassBase::ArithExpandOpsPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    arith::populateArithExpandOpsPatterns(patterns);

    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<vector::VectorDialect>();

    // clang-format off
    target.addIllegalOp<
      arith::CeilDivSIOp,
      arith::CeilDivUIOp,
      arith::FloorDivSIOp,
      arith::MaxSIOp,
      arith::MaxUIOp,
      arith::MinSIOp,
      arith::MinUIOp,
      arith::MaximumFOp,
      arith::MinimumFOp,
      arith::MaxNumFOp,
      arith::MinNumFOp,
      arith::ScalingExtFOp,
      arith::ScalingTruncFOp
    >();

    if (includeBf16)
      arith::populateExpandBFloat16Patterns(patterns);
    if (includeF8E8M0)
      arith::populateExpandF8E8M0Patterns(patterns);
    if (includeF4E2M1)
      arith::populateExpandF4E2M1Patterns(patterns);

    target.addDynamicallyLegalOp<arith::ExtFOp>(
      [=](arith::ExtFOp op) {
        Type inETy = getElementTypeOrSelf(op.getOperand().getType());
        Type outETy = getElementTypeOrSelf(op.getType());
        bool legalTypes = true;
        if (includeBf16)
          legalTypes &= !(inETy.isBF16() && outETy.isF32());
        if (includeF8E8M0)
          legalTypes &= !llvm::isa<Float8E8M0FNUType>(inETy);
        if (includeF4E2M1)
          legalTypes &= !llvm::isa<Float4E2M1FNType>(inETy);
        return legalTypes;
      });

    target.addDynamicallyLegalOp<arith::TruncFOp>(
      [=](arith::TruncFOp op)  {
        Type inETy = getElementTypeOrSelf(op.getOperand().getType());
        Type outETy = getElementTypeOrSelf(op.getType());
        bool legalTypes = true;
        if (includeBf16)
          legalTypes &= !(inETy.isF32() && outETy.isBF16());
        if (includeF8E8M0)
          legalTypes &= !(llvm::isa<Float8E8M0FNUType>(outETy));
        if (includeF4E2M1)
          legalTypes &= !llvm::isa<Float4E2M1FNType>(outETy);
        return legalTypes;
      });

    // clang-format on
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::arith::populateCeilFloorDivExpandOpsPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<CeilDivSIOpConverter, CeilDivUIOpConverter, FloorDivSIOpConverter>(
          patterns.getContext());
}

void mlir::arith::populateExpandBFloat16Patterns(RewritePatternSet &patterns) {
  patterns.add<BFloat16ExtFOpConverter, BFloat16TruncFOpConverter>(
      patterns.getContext());
}

void mlir::arith::populateExpandF4E2M1Patterns(RewritePatternSet &patterns) {
  patterns.add<F4E2M1ExtFOpConverter, F4E2M1TruncFOpConverter>(
      patterns.getContext());
}

void mlir::arith::populateExpandF8E8M0Patterns(RewritePatternSet &patterns) {
  patterns.add<F8E8M0ExtFOpConverter, F8E8M0TruncFOpConverter>(
      patterns.getContext());
}

void mlir::arith::populateExpandScalingExtTruncPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ScalingExtFOpConverter, ScalingTruncFOpConverter>(
      patterns.getContext());
}

void mlir::arith::populateArithExpandOpsPatterns(RewritePatternSet &patterns) {
  populateCeilFloorDivExpandOpsPatterns(patterns);
  populateExpandScalingExtTruncPatterns(patterns);
  // clang-format off
  patterns.add<
    MaxMinIOpConverter<MaxSIOp, arith::CmpIPredicate::sgt>,
    MaxMinIOpConverter<MaxUIOp, arith::CmpIPredicate::ugt>,
    MaxMinIOpConverter<MinSIOp, arith::CmpIPredicate::slt>,
    MaxMinIOpConverter<MinUIOp, arith::CmpIPredicate::ult>,
    MaximumMinimumFOpConverter<MaximumFOp, arith::CmpFPredicate::UGT>,
    MaximumMinimumFOpConverter<MinimumFOp, arith::CmpFPredicate::ULT>,
    MaxNumMinNumFOpConverter<MaxNumFOp, arith::CmpFPredicate::UGT>,
    MaxNumMinNumFOpConverter<MinNumFOp, arith::CmpFPredicate::ULT>
   >(patterns.getContext());
  // clang-format on
}
