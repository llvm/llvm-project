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

/// Expands CeilDivSIOp (n, m) into
///   1) x = (m > 0) ? -1 : 1
///   2) (n*m>0) ? ((n+x) / m) + 1 : - (-n / m)
struct CeilDivSIOpConverter : public OpRewritePattern<arith::CeilDivSIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::CeilDivSIOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type type = op.getType();
    Value a = op.getLhs();
    Value b = op.getRhs();
    Value plusOne = createConst(loc, type, 1, rewriter);
    Value zero = createConst(loc, type, 0, rewriter);
    Value minusOne = createConst(loc, type, -1, rewriter);
    // Compute x = (b>0) ? -1 : 1.
    Value compare =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, b, zero);
    Value x = rewriter.create<arith::SelectOp>(loc, compare, minusOne, plusOne);
    // Compute positive res: 1 + ((x+a)/b).
    Value xPlusA = rewriter.create<arith::AddIOp>(loc, x, a);
    Value xPlusADivB = rewriter.create<arith::DivSIOp>(loc, xPlusA, b);
    Value posRes = rewriter.create<arith::AddIOp>(loc, plusOne, xPlusADivB);
    // Compute negative res: - ((-a)/b).
    Value minusA = rewriter.create<arith::SubIOp>(loc, zero, a);
    Value minusADivB = rewriter.create<arith::DivSIOp>(loc, minusA, b);
    Value negRes = rewriter.create<arith::SubIOp>(loc, zero, minusADivB);
    // Result is (a*b>0) ? pos result : neg result.
    // Note, we want to avoid using a*b because of possible overflow.
    // The case that matters are a>0, a==0, a<0, b>0 and b<0. We do
    // not particuliarly care if a*b<0 is true or false when b is zero
    // as this will result in an illegal divide. So `a*b<0` can be reformulated
    // as `(a<0 && b<0) || (a>0 && b>0)' or `(a<0 && b<0) || (a>0 && b>=0)'.
    // We pick the first expression here.
    Value aNeg =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, a, zero);
    Value aPos =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, a, zero);
    Value bNeg =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, b, zero);
    Value bPos =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, b, zero);
    Value firstTerm = rewriter.create<arith::AndIOp>(loc, aNeg, bNeg);
    Value secondTerm = rewriter.create<arith::AndIOp>(loc, aPos, bPos);
    Value compareRes =
        rewriter.create<arith::OrIOp>(loc, firstTerm, secondTerm);
    // Perform substitution and return success.
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, compareRes, posRes,
                                                 negRes);
    return success();
  }
};

/// Expands FloorDivSIOp (n, m) into
///   1)  x = (m<0) ? 1 : -1
///   2)  return (n*m<0) ? - ((-n+x) / m) -1 : n / m
struct FloorDivSIOpConverter : public OpRewritePattern<arith::FloorDivSIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::FloorDivSIOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type type = op.getType();
    Value a = op.getLhs();
    Value b = op.getRhs();
    Value plusOne = createConst(loc, type, 1, rewriter);
    Value zero = createConst(loc, type, 0, rewriter);
    Value minusOne = createConst(loc, type, -1, rewriter);
    // Compute x = (b<0) ? 1 : -1.
    Value compare =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, b, zero);
    Value x = rewriter.create<arith::SelectOp>(loc, compare, plusOne, minusOne);
    // Compute negative res: -1 - ((x-a)/b).
    Value xMinusA = rewriter.create<arith::SubIOp>(loc, x, a);
    Value xMinusADivB = rewriter.create<arith::DivSIOp>(loc, xMinusA, b);
    Value negRes = rewriter.create<arith::SubIOp>(loc, minusOne, xMinusADivB);
    // Compute positive res: a/b.
    Value posRes = rewriter.create<arith::DivSIOp>(loc, a, b);
    // Result is (a*b<0) ? negative result : positive result.
    // Note, we want to avoid using a*b because of possible overflow.
    // The case that matters are a>0, a==0, a<0, b>0 and b<0. We do
    // not particuliarly care if a*b<0 is true or false when b is zero
    // as this will result in an illegal divide. So `a*b<0` can be reformulated
    // as `(a>0 && b<0) || (a>0 && b<0)' or `(a>0 && b<0) || (a>0 && b<=0)'.
    // We pick the first expression here.
    Value aNeg =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, a, zero);
    Value aPos =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, a, zero);
    Value bNeg =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, b, zero);
    Value bPos =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, b, zero);
    Value firstTerm = rewriter.create<arith::AndIOp>(loc, aNeg, bPos);
    Value secondTerm = rewriter.create<arith::AndIOp>(loc, aPos, bNeg);
    Value compareRes =
        rewriter.create<arith::OrIOp>(loc, firstTerm, secondTerm);
    // Perform substitution and return success.
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, compareRes, negRes,
                                                 posRes);
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

    Type i16Ty = b.getI16Type();
    Type i32Ty = b.getI32Type();
    if (auto shapedTy = dyn_cast<ShapedType>(operandTy)) {
      i16Ty = shapedTy.clone(i16Ty);
      i32Ty = shapedTy.clone(i32Ty);
    }

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

    Type i1Ty = b.getI1Type();
    Type i16Ty = b.getI16Type();
    Type i32Ty = b.getI32Type();
    Type f32Ty = b.getF32Type();
    if (auto shapedTy = dyn_cast<ShapedType>(operandTy)) {
      i1Ty = shapedTy.clone(i1Ty);
      i16Ty = shapedTy.clone(i16Ty);
      i32Ty = shapedTy.clone(i32Ty);
      f32Ty = shapedTy.clone(f32Ty);
    }

    Value bitcast = b.create<arith::BitcastOp>(i32Ty, operand);

    Value c23 = createConst(op.getLoc(), i32Ty, 23, rewriter);
    Value c31 = createConst(op.getLoc(), i32Ty, 31, rewriter);
    Value c23Mask = createConst(op.getLoc(), i32Ty, (1 << 23) - 1, rewriter);
    Value expMask =
        createConst(op.getLoc(), i32Ty, ((1 << 8) - 1) << 23, rewriter);
    Value expMax =
        createConst(op.getLoc(), i32Ty, ((1 << 8) - 2) << 23, rewriter);

    // Grab the sign bit.
    Value sign = b.create<arith::ShRUIOp>(bitcast, c31);

    // Our mantissa rounding value depends on the sign bit and the last
    // truncated bit.
    Value cManRound = createConst(op.getLoc(), i32Ty, (1 << 15), rewriter);
    cManRound = b.create<arith::SubIOp>(cManRound, sign);

    // Grab out the mantissa and directly apply rounding.
    Value man = b.create<arith::AndIOp>(bitcast, c23Mask);
    Value manRound = b.create<arith::AddIOp>(man, cManRound);

    // Grab the overflow bit and shift right if we overflow.
    Value roundBit = b.create<arith::ShRUIOp>(manRound, c23);
    Value manNew = b.create<arith::ShRUIOp>(manRound, roundBit);

    // Grab the exponent and round using the mantissa's carry bit.
    Value exp = b.create<arith::AndIOp>(bitcast, expMask);
    Value expCarry = b.create<arith::AddIOp>(exp, manRound);
    expCarry = b.create<arith::AndIOp>(expCarry, expMask);

    // If the exponent is saturated, we keep the max value.
    Value expCmp =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::uge, exp, expMax);
    exp = b.create<arith::SelectOp>(expCmp, exp, expCarry);

    // If the exponent is max and we rolled over, keep the old mantissa.
    Value roundBitBool = b.create<arith::TruncIOp>(i1Ty, roundBit);
    Value keepOldMan = b.create<arith::AndIOp>(expCmp, roundBitBool);
    man = b.create<arith::SelectOp>(keepOldMan, man, manNew);

    // Assemble the now rounded f32 value (as an i32).
    Value rounded = b.create<arith::ShLIOp>(sign, c31);
    rounded = b.create<arith::OrIOp>(rounded, exp);
    rounded = b.create<arith::OrIOp>(rounded, man);

    Value c16 = createConst(op.getLoc(), i32Ty, 16, rewriter);
    Value shr = b.create<arith::ShRUIOp>(rounded, c16);
    Value trunc = b.create<arith::TruncIOp>(i16Ty, shr);
    Value result = b.create<arith::BitcastOp>(resultTy, trunc);

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
      arith::MaximumFOp,
      arith::MinimumFOp,
      arith::MaxNumFOp,
      arith::MinNumFOp
    >();

    if (includeBf16) {
      arith::populateExpandBFloat16Patterns(patterns);
      target.addDynamicallyLegalOp<arith::ExtFOp>(
        [](arith::ExtFOp op) {
          Type inETy = getElementTypeOrSelf(op.getOperand().getType());
          Type outETy = getElementTypeOrSelf(op.getType());
          return !(inETy.isBF16() && outETy.isF32());
        });

      target.addDynamicallyLegalOp<arith::TruncFOp>(
        [](arith::TruncFOp op)  {
          Type inETy = getElementTypeOrSelf(op.getOperand().getType());
          Type outETy = getElementTypeOrSelf(op.getType());
          return !(inETy.isF32() && outETy.isBF16());
        });
    }

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

void mlir::arith::populateArithExpandOpsPatterns(RewritePatternSet &patterns) {
  populateCeilFloorDivExpandOpsPatterns(patterns);
  // clang-format off
  patterns.add<
    MaximumMinimumFOpConverter<MaximumFOp, arith::CmpFPredicate::UGT>,
    MaximumMinimumFOpConverter<MinimumFOp, arith::CmpFPredicate::ULT>,
    MaxNumMinNumFOpConverter<MaxNumFOp, arith::CmpFPredicate::UGT>,
    MaxNumMinNumFOpConverter<MinNumFOp, arith::CmpFPredicate::ULT>
   >(patterns.getContext());
  // clang-format on
}
