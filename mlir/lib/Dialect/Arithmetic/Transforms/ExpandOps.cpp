//===- ExpandOps.cpp - Pass to legalize Arithmetic ops for LLVM lowering --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"

using namespace mlir;

namespace {

/// Expands CeilDivSIOp (n, m) into
///   1) x = (m > 0) ? -1 : 1
///   2) (n*m>0) ? ((n+x) / m) + 1 : - (-n / m)
struct CeilDivSIOpConverter : public OpRewritePattern<arith::CeilDivSIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::CeilDivSIOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto signedCeilDivIOp = cast<arith::CeilDivSIOp>(op);
    Type type = signedCeilDivIOp.getType();
    Value a = signedCeilDivIOp.lhs();
    Value b = signedCeilDivIOp.rhs();
    Value plusOne = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(type, 1));
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(type, 0));
    Value minusOne = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(type, -1));
    // Compute x = (b>0) ? -1 : 1.
    Value compare =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, b, zero);
    Value x = rewriter.create<SelectOp>(loc, compare, minusOne, plusOne);
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
    Value res = rewriter.create<SelectOp>(loc, compareRes, posRes, negRes);
    // Perform substitution and return success.
    rewriter.replaceOp(op, {res});
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
    arith::FloorDivSIOp signedFloorDivIOp = cast<arith::FloorDivSIOp>(op);
    Type type = signedFloorDivIOp.getType();
    Value a = signedFloorDivIOp.lhs();
    Value b = signedFloorDivIOp.rhs();
    Value plusOne = rewriter.create<arith::ConstantIntOp>(loc, 1, type);
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, type);
    Value minusOne = rewriter.create<arith::ConstantIntOp>(loc, -1, type);
    // Compute x = (b<0) ? 1 : -1.
    Value compare =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, b, zero);
    Value x = rewriter.create<SelectOp>(loc, compare, plusOne, minusOne);
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
    Value res = rewriter.create<SelectOp>(loc, compareRes, negRes, posRes);
    // Perform substitution and return success.
    rewriter.replaceOp(op, {res});
    return success();
  }
};

struct ArithmeticExpandOpsPass
    : public ArithmeticExpandOpsBase<ArithmeticExpandOpsPass> {
  void runOnFunction() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    arith::populateArithmeticExpandOpsPatterns(patterns);

    target.addLegalDialect<arith::ArithmeticDialect, StandardOpsDialect>();
    target.addIllegalOp<arith::CeilDivSIOp, arith::FloorDivSIOp>();

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // end anonymous namespace

void mlir::arith::populateArithmeticExpandOpsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<CeilDivSIOpConverter, FloorDivSIOpConverter>(
      patterns.getContext());
}

std::unique_ptr<Pass> mlir::arith::createArithmeticExpandOpsPass() {
  return std::make_unique<ArithmeticExpandOpsPass>();
}
