//===- ExpandOps.cpp - Pass to legalize Arith ops for LLVM lowering --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
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
    return rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(shapedTy, attr));
  }

  return rewriter.create<arith::ConstantOp>(loc, attr);
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
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::CeilDivUIOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value a = op.getLhs();
    Value b = op.getRhs();
    Value zero = createConst(loc, a.getType(), 0, rewriter);
    Value compare =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, a, zero);
    Value one = createConst(loc, a.getType(), 1, rewriter);
    Value minusOne = rewriter.create<arith::SubIOp>(loc, a, one);
    Value quotient = rewriter.create<arith::DivUIOp>(loc, minusOne, b);
    Value plusOne = rewriter.create<arith::AddIOp>(loc, quotient, one);
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
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::CeilDivSIOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type type = op.getType();
    Value a = op.getLhs();
    Value b = op.getRhs();

    Value zero = createConst(loc, type, 0, rewriter);
    Value one = createConst(loc, type, 1, rewriter);

    Value quotient = rewriter.create<arith::DivSIOp>(loc, a, b);
    Value product = rewriter.create<arith::MulIOp>(loc, quotient, b);
    Value notEqualDivisor = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, a, product);

    Value aNeg =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, a, zero);
    Value bNeg =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, b, zero);

    Value signEqual = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, aNeg, bNeg);
    Value cond =
        rewriter.create<arith::AndIOp>(loc, notEqualDivisor, signEqual);

    Value quotientPlusOne = rewriter.create<arith::AddIOp>(loc, quotient, one);

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
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::FloorDivSIOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type type = op.getType();
    Value a = op.getLhs();
    Value b = op.getRhs();

    Value quotient = rewriter.create<arith::DivSIOp>(loc, a, b);
    Value product = rewriter.create<arith::MulIOp>(loc, quotient, b);
    Value notEqualDivisor = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, a, product);
    Value zero = createConst(loc, type, 0, rewriter);

    Value aNeg =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, a, zero);
    Value bNeg =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, b, zero);

    Value signOpposite = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, aNeg, bNeg);
    Value cond =
        rewriter.create<arith::AndIOp>(loc, notEqualDivisor, signOpposite);

    Value minusOne = createConst(loc, type, -1, rewriter);
    Value quotientMinusOne =
        rewriter.create<arith::AddIOp>(loc, quotient, minusOne);

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

    Value cmp = rewriter.create<arith::CmpIOp>(op.getLoc(), pred, lhs, rhs);
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
    Value cmp = rewriter.create<arith::CmpFOp>(loc, pred, lhs, rhs);
    Value select = rewriter.create<arith::SelectOp>(loc, cmp, lhs, rhs);

    // Handle the case where rhs is NaN: 'isNaN(rhs) ? rhs : select'.
    Value isNaN = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNO,
                                                 rhs, rhs);
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
    Value cmp = rewriter.create<arith::CmpFOp>(loc, pred, lhs, rhs);
    Value select = rewriter.create<arith::SelectOp>(loc, cmp, lhs, rhs);

    // Handle the case where lhs is NaN: 'isNaN(lhs) ? rhs : select'.
    Value isNaN = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNO,
                                                 lhs, lhs);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isNaN, rhs, select);
    return success();
  }
};

struct BFloat16ExtFOpConverter : public OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;
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

    Value bitcast = b.create<arith::BitcastOp>(i16Ty, operand);
    Value exti = b.create<arith::ExtUIOp>(i32Ty, bitcast);

    Value c16 = createConst(op.getLoc(), i32Ty, 16, rewriter);
    Value shl = b.create<arith::ShLIOp>(exti, c16);
    Value result = b.create<arith::BitcastOp>(resultTy, shl);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct BFloat16TruncFOpConverter : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;
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
        b.create<arith::CmpFOp>(arith::CmpFPredicate::UNE, operand, operand);
    // Constant used to make the rounding bias.
    Value c7FFF = createConst(op.getLoc(), i32Ty, 0x7fff, rewriter);
    // Constant used to generate a quiet NaN.
    Value c7FC0I16 = createConst(op.getLoc(), i16Ty, 0x7fc0, rewriter);
    // Small constants used to address bits.
    Value c16 = createConst(op.getLoc(), i32Ty, 16, rewriter);
    Value c1 = createConst(op.getLoc(), i32Ty, 1, rewriter);
    // Reinterpret the input f32 value as bits.
    Value bitcast = b.create<arith::BitcastOp>(i32Ty, operand);
    // Read bit 16 as a value in {0,1}.
    Value bit16 =
        b.create<arith::AndIOp>(b.create<arith::ShRUIOp>(bitcast, c16), c1);
    // Determine the rounding bias to add as either 0x7fff or 0x8000 depending
    // on bit 16, implementing the tie-breaking "to nearest even".
    Value roundingBias = b.create<arith::AddIOp>(bit16, c7FFF);
    // Add the rounding bias. Generally we want this to be added to the
    // mantissa, but nothing prevents this to from carrying into the exponent
    // bits, which would feel like a bug, but this is the magic trick here:
    // when that happens, the mantissa gets reset to zero and the exponent
    // gets incremented by the carry... which is actually exactly what we
    // want.
    Value biased = b.create<arith::AddIOp>(bitcast, roundingBias);
    // Now that the rounding-bias has been added, truncating the low bits
    // yields the correctly rounded result.
    Value biasedAndShifted = b.create<arith::ShRUIOp>(biased, c16);
    Value normalCaseResultI16 =
        b.create<arith::TruncIOp>(i16Ty, biasedAndShifted);
    // Select either the above-computed result, or a quiet NaN constant
    // if the input was NaN.
    Value select =
        b.create<arith::SelectOp>(isNan, c7FC0I16, normalCaseResultI16);
    Value result = b.create<arith::BitcastOp>(resultTy, select);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct F8E8M0ExtFOpConverter : public OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;
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

    Value bitcast = b.create<arith::BitcastOp>(i8Ty, operand);
    // create constants for NaNs
    Value cF8NaN = createConst(op.getLoc(), i8Ty, 0xff, rewriter);
    Value cF32NaN = createConst(op.getLoc(), i32Ty, 0xffffffff, rewriter);
    Value cF32MantissaWidth = createConst(op->getLoc(), i32Ty, 23, rewriter);

    Value exti = b.create<arith::ExtUIOp>(i32Ty, bitcast);
    Value f32Bits = b.create<arith::ShLIOp>(exti, cF32MantissaWidth);

    Value isNan =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, bitcast, cF8NaN);
    // select for NaNs
    f32Bits = b.create<arith::SelectOp>(isNan, cF32NaN, f32Bits);
    Value result = b.create<arith::BitcastOp>(f32Ty, f32Bits);
    if (resultETy.getIntOrFloatBitWidth() < 32) {
      result = b.create<arith::TruncFOp>(resultTy, result);
    } else if (resultETy.getIntOrFloatBitWidth() > 32) {
      result = b.create<arith::ExtFOp>(resultTy, result);
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
  using OpRewritePattern::OpRewritePattern;
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
      operand = b.create<arith::ExtFOp>(f32Ty, operand);
    } else if (operandETy.getIntOrFloatBitWidth() > 32) {
      operand = b.create<arith::TruncFOp>(f32Ty, operand);
    }
    Value f32Bits = b.create<arith::BitcastOp>(i32Ty, operand);
    Value cF32MantissaWidth = createConst(op->getLoc(), i32Ty, 23, rewriter);
    Value f32SignExp = b.create<arith::ShRUIOp>(f32Bits, cF32MantissaWidth);
    Value exp8Bits = b.create<arith::TruncIOp>(i8Ty, f32SignExp);
    Value result = b.create<arith::BitcastOp>(resultTy, exp8Bits);
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

    arith::populateArithExpandOpsPatterns(patterns);

    target.addLegalDialect<arith::ArithDialect>();
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
      arith::MinNumFOp
    >();

    if (includeBf16) {
      arith::populateExpandBFloat16Patterns(patterns);
    }
    if (includeF8E8M0) {
      arith::populateExpandF8E8M0Patterns(patterns);
    }

    target.addDynamicallyLegalOp<arith::ExtFOp>(
      [=](arith::ExtFOp op) {
        Type inETy = getElementTypeOrSelf(op.getOperand().getType());
        Type outETy = getElementTypeOrSelf(op.getType());
        bool legalTypes = true;
        if (includeBf16) 
          legalTypes &= !(inETy.isBF16() && outETy.isF32());
        if (includeF8E8M0)
          legalTypes &= !llvm::isa<Float8E8M0FNUType>(inETy);
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

void mlir::arith::populateExpandF8E8M0Patterns(RewritePatternSet &patterns) {
  patterns.add<F8E8M0ExtFOpConverter, F8E8M0TruncFOpConverter>(
      patterns.getContext());
}

void mlir::arith::populateArithExpandOpsPatterns(RewritePatternSet &patterns) {
  populateCeilFloorDivExpandOpsPatterns(patterns);
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
