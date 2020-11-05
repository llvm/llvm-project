//===- StdExpandDivs.cpp - Code to prepare Std for lowring Divs 0to LLVM  -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file Std transformations to expand Divs operation to help for the
// lowering to LLVM. Currently implemented tranformations are Ceil and Floor
// for Signed Integers.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {

/// Expands SignedCeilDivIOP (n, m) into
///   1) x = (m > 0) ? -1 : 1
///   2) (n*m>0) ? ((n+x) / m) + 1 : - (-n / m)
struct SignedCeilDivIOpConverter : public OpRewritePattern<SignedCeilDivIOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SignedCeilDivIOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    SignedCeilDivIOp signedCeilDivIOp = cast<SignedCeilDivIOp>(op);
    Type type = signedCeilDivIOp.getType();
    Value a = signedCeilDivIOp.lhs();
    Value b = signedCeilDivIOp.rhs();
    Value plusOne =
        rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(type, 1));
    Value zero =
        rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(type, 0));
    Value minusOne =
        rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(type, -1));
    // Compute x = (b>0) ? -1 : 1.
    Value compare = rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt, b, zero);
    Value x = rewriter.create<SelectOp>(loc, compare, minusOne, plusOne);
    // Compute positive res: 1 + ((x+a)/b).
    Value xPlusA = rewriter.create<AddIOp>(loc, x, a);
    Value xPlusADivB = rewriter.create<SignedDivIOp>(loc, xPlusA, b);
    Value posRes = rewriter.create<AddIOp>(loc, plusOne, xPlusADivB);
    // Compute negative res: - ((-a)/b).
    Value minusA = rewriter.create<SubIOp>(loc, zero, a);
    Value minusADivB = rewriter.create<SignedDivIOp>(loc, minusA, b);
    Value negRes = rewriter.create<SubIOp>(loc, zero, minusADivB);
    // Result is (a*b>0) ? pos result : neg result.
    // Note, we want to avoid using a*b because of possible overflow.
    // The case that matters are a>0, a==0, a<0, b>0 and b<0. We do
    // not particuliarly care if a*b<0 is true or false when b is zero
    // as this will result in an illegal divide. So `a*b<0` can be reformulated
    // as `(a<0 && b<0) || (a>0 && b>0)' or `(a<0 && b<0) || (a>0 && b>=0)'.
    // We pick the first expression here.
    Value aNeg = rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, a, zero);
    Value aPos = rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt, a, zero);
    Value bNeg = rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, b, zero);
    Value bPos = rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt, b, zero);
    Value firstTerm = rewriter.create<AndOp>(loc, aNeg, bNeg);
    Value secondTerm = rewriter.create<AndOp>(loc, aPos, bPos);
    Value compareRes = rewriter.create<OrOp>(loc, firstTerm, secondTerm);
    Value res = rewriter.create<SelectOp>(loc, compareRes, posRes, negRes);
    // Perform substitution and return success.
    rewriter.replaceOp(op, {res});
    return success();
  }
};

/// Expands SignedFloorDivIOP (n, m) into
///   1)  x = (m<0) ? 1 : -1
///   2)  return (n*m<0) ? - ((-n+x) / m) -1 : n / m
struct SignedFloorDivIOpConverter : public OpRewritePattern<SignedFloorDivIOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SignedFloorDivIOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    SignedFloorDivIOp signedFloorDivIOp = cast<SignedFloorDivIOp>(op);
    Type type = signedFloorDivIOp.getType();
    Value a = signedFloorDivIOp.lhs();
    Value b = signedFloorDivIOp.rhs();
    Value plusOne =
        rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(type, 1));
    Value zero =
        rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(type, 0));
    Value minusOne =
        rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(type, -1));
    // Compute x = (b<0) ? 1 : -1.
    Value compare = rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, b, zero);
    Value x = rewriter.create<SelectOp>(loc, compare, plusOne, minusOne);
    // Compute negative res: -1 - ((x-a)/b).
    Value xMinusA = rewriter.create<SubIOp>(loc, x, a);
    Value xMinusADivB = rewriter.create<SignedDivIOp>(loc, xMinusA, b);
    Value negRes = rewriter.create<SubIOp>(loc, minusOne, xMinusADivB);
    // Compute positive res: a/b.
    Value posRes = rewriter.create<SignedDivIOp>(loc, a, b);
    // Result is (a*b<0) ? negative result : positive result.
    // Note, we want to avoid using a*b because of possible overflow.
    // The case that matters are a>0, a==0, a<0, b>0 and b<0. We do
    // not particuliarly care if a*b<0 is true or false when b is zero
    // as this will result in an illegal divide. So `a*b<0` can be reformulated
    // as `(a>0 && b<0) || (a>0 && b<0)' or `(a>0 && b<0) || (a>0 && b<=0)'.
    // We pick the first expression here.
    Value aNeg = rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, a, zero);
    Value aPos = rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt, a, zero);
    Value bNeg = rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, b, zero);
    Value bPos = rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt, b, zero);
    Value firstTerm = rewriter.create<AndOp>(loc, aNeg, bPos);
    Value secondTerm = rewriter.create<AndOp>(loc, aPos, bNeg);
    Value compareRes = rewriter.create<OrOp>(loc, firstTerm, secondTerm);
    Value res = rewriter.create<SelectOp>(loc, compareRes, negRes, posRes);
    // Perform substitution and return success.
    rewriter.replaceOp(op, {res});
    return success();
  }
};

} // namespace

namespace {
struct StdExpandDivs : public StdExpandDivsBase<StdExpandDivs> {
  void runOnFunction() override;
};
} // namespace

void StdExpandDivs::runOnFunction() {
  MLIRContext &ctx = getContext();

  OwningRewritePatternList patterns;
  populateStdExpandDivsRewritePatterns(&ctx, patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<StandardOpsDialect>();
  target.addIllegalOp<SignedCeilDivIOp>();
  target.addIllegalOp<SignedFloorDivIOp>();
  if (failed(
          applyPartialConversion(getFunction(), target, std::move(patterns))))
    signalPassFailure();
}

void mlir::populateStdExpandDivsRewritePatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<SignedCeilDivIOpConverter, SignedFloorDivIOpConverter>(
      context);
}

std::unique_ptr<Pass> mlir::createStdExpandDivsPass() {
  return std::make_unique<StdExpandDivs>();
}
