//===- PolynomialApproximation.cpp - Approximate math operations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements expansion of math operations to fast approximations
// that do not rely on any of the library functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;

static bool isValidFloatType(Type type) {
  if (auto vectorType = type.dyn_cast<VectorType>())
    return vectorType.getElementType().isa<FloatType>();
  return type.isa<FloatType>();
}

//----------------------------------------------------------------------------//
// A PatternRewriter wrapper that provides concise API for building expansions
// for operations on float scalars or vectors.
//----------------------------------------------------------------------------//

namespace {
class FloatApproximationBuilder {
public:
  FloatApproximationBuilder(Location loc, Type type, PatternRewriter &rewriter);

  Value constant(double value) const;

  Value abs(Value a) const;
  Value min(Value a, Value b) const;
  Value max(Value a, Value b) const;
  Value mul(Value a, Value b) const;
  Value div(Value a, Value b) const;

  // Fused multiple-add operation: a * b + c.
  Value madd(Value a, Value b, Value c) const;

  // Compares values `a` and `b` with the given `predicate`.
  Value cmp(CmpFPredicate predicate, Value a, Value b) const;

  // Selects values from `a` or `b` based on the `predicate`.
  Value select(Value predicate, Value a, Value b) const;

private:
  Location loc;
  PatternRewriter &rewriter;
  VectorType vectorType; // can be null for scalar type
  FloatType elementType;
};
} // namespace

FloatApproximationBuilder::FloatApproximationBuilder(Location loc, Type type,
                                                     PatternRewriter &rewriter)
    : loc(loc), rewriter(rewriter) {
  vectorType = type.dyn_cast<VectorType>();

  if (vectorType)
    elementType = vectorType.getElementType().cast<FloatType>();
  else
    elementType = type.cast<FloatType>();
}

Value FloatApproximationBuilder::constant(double value) const {
  auto attr = rewriter.getFloatAttr(elementType, value);
  Value scalar = rewriter.create<ConstantOp>(loc, attr);

  if (vectorType)
    return rewriter.create<BroadcastOp>(loc, vectorType, scalar);
  return scalar;
}

Value FloatApproximationBuilder::abs(Value a) const {
  return rewriter.create<AbsFOp>(loc, a);
}

Value FloatApproximationBuilder::min(Value a, Value b) const {
  return select(cmp(CmpFPredicate::OLT, a, b), a, b);
}
Value FloatApproximationBuilder::max(Value a, Value b) const {
  return select(cmp(CmpFPredicate::OGT, a, b), a, b);
}
Value FloatApproximationBuilder::mul(Value a, Value b) const {
  return rewriter.create<MulFOp>(loc, a, b);
}

Value FloatApproximationBuilder::div(Value a, Value b) const {
  return rewriter.create<DivFOp>(loc, a, b);
}

Value FloatApproximationBuilder::madd(Value a, Value b, Value c) const {
  return rewriter.create<FmaFOp>(loc, a, b, c);
}

Value FloatApproximationBuilder::cmp(CmpFPredicate predicate, Value a,
                                     Value b) const {
  return rewriter.create<CmpFOp>(loc, predicate, a, b);
}

Value FloatApproximationBuilder::select(Value predicate, Value a,
                                        Value b) const {
  return rewriter.create<SelectOp>(loc, predicate, a, b);
}

//----------------------------------------------------------------------------//
// TanhOp approximation.
//----------------------------------------------------------------------------//

namespace {
struct TanhApproximation : public OpRewritePattern<math::TanhOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::TanhOp op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
TanhApproximation::matchAndRewrite(math::TanhOp op,
                                   PatternRewriter &rewriter) const {
  if (!isValidFloatType(op.operand().getType()))
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  Value operand = op.operand();
  FloatApproximationBuilder builder(op->getLoc(), operand.getType(), rewriter);

  // Clamp operand into [plusClamp, minusClamp] range.
  Value plusClamp = builder.constant(7.90531110763549805);
  Value minusClamp = builder.constant(-7.9053111076354980);
  Value x = builder.max(builder.min(operand, plusClamp), minusClamp);

  // Mask for tiny values that are approximated with `operand`.
  Value tiny = builder.constant(0.0004f);
  Value tinyMask = builder.cmp(CmpFPredicate::OLT, builder.abs(operand), tiny);

  // The monomial coefficients of the numerator polynomial (odd).
  Value alpha1 = builder.constant(4.89352455891786e-03);
  Value alpha3 = builder.constant(6.37261928875436e-04);
  Value alpha5 = builder.constant(1.48572235717979e-05);
  Value alpha7 = builder.constant(5.12229709037114e-08);
  Value alpha9 = builder.constant(-8.60467152213735e-11);
  Value alpha11 = builder.constant(2.00018790482477e-13);
  Value alpha13 = builder.constant(-2.76076847742355e-16);

  // The monomial coefficients of the denominator polynomial (even).
  Value beta0 = builder.constant(4.89352518554385e-03);
  Value beta2 = builder.constant(2.26843463243900e-03);
  Value beta4 = builder.constant(1.18534705686654e-04);
  Value beta6 = builder.constant(1.19825839466702e-06);

  // Since the polynomials are odd/even, we need x^2.
  Value x2 = builder.mul(x, x);

  // Evaluate the numerator polynomial p.
  Value p = builder.madd(x2, alpha13, alpha11);
  p = builder.madd(x2, p, alpha9);
  p = builder.madd(x2, p, alpha7);
  p = builder.madd(x2, p, alpha5);
  p = builder.madd(x2, p, alpha3);
  p = builder.madd(x2, p, alpha1);
  p = builder.mul(x, p);

  // Evaluate the denominator polynomial q.
  Value q = builder.madd(x2, beta6, beta4);
  q = builder.madd(x2, q, beta2);
  q = builder.madd(x2, q, beta0);

  // Divide the numerator by the denominator.
  Value res = builder.select(tinyMask, x, builder.div(p, q));

  rewriter.replaceOp(op, res);

  return success();
}

//----------------------------------------------------------------------------//

void mlir::populateMathPolynomialApproximationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<TanhApproximation>(ctx);
}
