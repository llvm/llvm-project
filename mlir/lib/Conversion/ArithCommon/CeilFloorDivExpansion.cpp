//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithCommon/CeilFloorDivExpansion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

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
} // namespace

void mlir::arith::populateCeilFloorDivExpandOpsPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<CeilDivSIOpConverter, CeilDivUIOpConverter, FloorDivSIOpConverter>(
          patterns.getContext());
}
