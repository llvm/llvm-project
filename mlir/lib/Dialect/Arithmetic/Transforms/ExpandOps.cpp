//===- ExpandOps.cpp - Pass to legalize Arithmetic ops for LLVM lowering --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace arith {
#define GEN_PASS_DEF_ARITHMETICEXPANDOPSPASS
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h.inc"
} // namespace arith
} // namespace mlir

using namespace mlir;

/// Create an integer or index constant.
static Value createConst(Location loc, Type type, int value,
                         PatternRewriter &rewriter) {
  return rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(type, value));
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
struct MaxMinFOpConverter : public OpRewritePattern<OpTy> {
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

template <typename OpTy, arith::CmpIPredicate pred>
struct MaxMinIOpConverter : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    Location loc = op.getLoc();
    Value cmp = rewriter.create<arith::CmpIOp>(loc, pred, lhs, rhs);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, cmp, lhs, rhs);
    return success();
  }
};

struct ArithmeticExpandOpsPass
    : public arith::impl::ArithmeticExpandOpsPassBase<ArithmeticExpandOpsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    arith::populateArithmeticExpandOpsPatterns(patterns);

    target.addLegalDialect<arith::ArithmeticDialect>();
    // clang-format off
    target.addIllegalOp<
      arith::CeilDivSIOp,
      arith::CeilDivUIOp,
      arith::FloorDivSIOp,
      arith::MaxFOp,
      arith::MaxSIOp,
      arith::MaxUIOp,
      arith::MinFOp,
      arith::MinSIOp,
      arith::MinUIOp
    >();
    // clang-format on
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::arith::populateArithmeticExpandOpsPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    CeilDivSIOpConverter,
    CeilDivUIOpConverter,
    FloorDivSIOpConverter,
    MaxMinFOpConverter<MaxFOp, arith::CmpFPredicate::UGT>,
    MaxMinFOpConverter<MinFOp, arith::CmpFPredicate::ULT>,
    MaxMinIOpConverter<MaxSIOp, arith::CmpIPredicate::sgt>,
    MaxMinIOpConverter<MaxUIOp, arith::CmpIPredicate::ugt>,
    MaxMinIOpConverter<MinSIOp, arith::CmpIPredicate::slt>,
    MaxMinIOpConverter<MinUIOp, arith::CmpIPredicate::ult>
   >(patterns.getContext());
  // clang-format on
}

std::unique_ptr<Pass> mlir::arith::createArithmeticExpandOpsPass() {
  return std::make_unique<ArithmeticExpandOpsPass>();
}
