//===- InferIntRangeInterfaceTest.cpp - Unit Tests for InferIntRange... --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

#include <gtest/gtest.h>

using namespace mlir;

namespace {

using mlir::intrange::OverflowFlags;

template <typename OpTy>
ConstantIntRanges inferBinaryOpResult(OpTy op,
                                      ArrayRef<ConstantIntRanges> argRanges) {
  std::optional<ConstantIntRanges> inferred;
  op.inferResultRanges(argRanges, [&](Value v, const ConstantIntRanges &range) {
    if (v == op.getResult())
      inferred = range;
  });
  assert(inferred.has_value() && "binary op did not produce a result range");
  return *inferred;
}

} // namespace

TEST(IntRangeAttrs, BasicConstructors) {
  APInt zero = APInt::getZero(64);
  APInt two(64, 2);
  APInt three(64, 3);
  ConstantIntRanges boundedAbove(zero, two, zero, three);
  EXPECT_EQ(boundedAbove.umin(), zero);
  EXPECT_EQ(boundedAbove.umax(), two);
  EXPECT_EQ(boundedAbove.smin(), zero);
  EXPECT_EQ(boundedAbove.smax(), three);
}

TEST(IntRangeAttrs, FromUnsigned) {
  APInt zero = APInt::getZero(64);
  APInt maxInt = APInt::getSignedMaxValue(64);
  APInt minInt = APInt::getSignedMinValue(64);
  APInt minIntPlusOne = minInt + 1;

  ConstantIntRanges canPortToSigned =
      ConstantIntRanges::fromUnsigned(zero, maxInt);
  EXPECT_EQ(canPortToSigned.smin(), zero);
  EXPECT_EQ(canPortToSigned.smax(), maxInt);

  ConstantIntRanges cantPortToSigned =
      ConstantIntRanges::fromUnsigned(zero, minInt);
  EXPECT_EQ(cantPortToSigned.smin(), minInt);
  EXPECT_EQ(cantPortToSigned.smax(), maxInt);

  ConstantIntRanges signedNegative =
      ConstantIntRanges::fromUnsigned(minInt, minIntPlusOne);
  EXPECT_EQ(signedNegative.smin(), minInt);
  EXPECT_EQ(signedNegative.smax(), minIntPlusOne);
}

TEST(IntRangeAttrs, FromSigned) {
  APInt zero = APInt::getZero(64);
  APInt one = zero + 1;
  APInt negOne = zero - 1;
  APInt intMax = APInt::getSignedMaxValue(64);
  APInt intMin = APInt::getSignedMinValue(64);
  APInt uintMax = APInt::getMaxValue(64);

  ConstantIntRanges noUnsignedBound =
      ConstantIntRanges::fromSigned(negOne, one);
  EXPECT_EQ(noUnsignedBound.umin(), zero);
  EXPECT_EQ(noUnsignedBound.umax(), uintMax);

  ConstantIntRanges positive = ConstantIntRanges::fromSigned(one, intMax);
  EXPECT_EQ(positive.umin(), one);
  EXPECT_EQ(positive.umax(), intMax);

  ConstantIntRanges negative = ConstantIntRanges::fromSigned(intMin, negOne);
  EXPECT_EQ(negative.umin(), intMin);
  EXPECT_EQ(negative.umax(), negOne);

  ConstantIntRanges preserved = ConstantIntRanges::fromSigned(zero, one);
  EXPECT_EQ(preserved.umin(), zero);
  EXPECT_EQ(preserved.umax(), one);
}

TEST(IntRangeAttrs, Join) {
  APInt zero = APInt::getZero(64);
  APInt one = zero + 1;
  APInt two = zero + 2;
  APInt intMin = APInt::getSignedMinValue(64);
  APInt intMax = APInt::getSignedMaxValue(64);
  APInt uintMax = APInt::getMaxValue(64);

  ConstantIntRanges maximal(zero, uintMax, intMin, intMax);
  ConstantIntRanges zeroOne(zero, one, zero, one);

  EXPECT_EQ(zeroOne.rangeUnion(maximal), maximal);
  EXPECT_EQ(maximal.rangeUnion(zeroOne), maximal);

  EXPECT_EQ(zeroOne.rangeUnion(zeroOne), zeroOne);

  ConstantIntRanges oneTwo(one, two, one, two);
  ConstantIntRanges zeroTwo(zero, two, zero, two);
  EXPECT_EQ(zeroOne.rangeUnion(oneTwo), zeroTwo);

  ConstantIntRanges zeroOneUnsignedOnly(zero, one, intMin, intMax);
  ConstantIntRanges zeroOneSignedOnly(zero, uintMax, zero, one);
  EXPECT_EQ(zeroOneUnsignedOnly.rangeUnion(zeroOneSignedOnly), maximal);
}

TEST(IntRangeAttrs, OverflowFlags) {
  APInt zero = APInt::getZero(64);
  APInt one = zero + 1;
  APInt two = zero + 2;

  ConstantIntRanges nswOnly(zero, one, zero, one, OverflowFlags::Nsw);
  ConstantIntRanges nuwOnly(one, two, one, two, OverflowFlags::Nuw);

  EXPECT_NE(nswOnly.getOverflowFlags() & OverflowFlags::Nsw,
            OverflowFlags::None);
  EXPECT_EQ(nswOnly.getOverflowFlags() & OverflowFlags::Nuw,
            OverflowFlags::None);

  ConstantIntRanges both =
      nswOnly.withOverflowFlags(OverflowFlags::Nsw | OverflowFlags::Nuw);
  EXPECT_NE(both.getOverflowFlags() & OverflowFlags::Nsw, OverflowFlags::None);
  EXPECT_NE(both.getOverflowFlags() & OverflowFlags::Nuw, OverflowFlags::None);

  // rangeUnion conservatively preserves only proofs present in both inputs.
  EXPECT_EQ(nswOnly.rangeUnion(nuwOnly).getOverflowFlags(),
            OverflowFlags::None);
  EXPECT_EQ(both.rangeUnion(nswOnly).getOverflowFlags(), OverflowFlags::Nsw);

  // intersection preserves proofs from either input.
  EXPECT_EQ(nswOnly.intersection(nuwOnly).getOverflowFlags(),
            OverflowFlags::Nsw | OverflowFlags::Nuw);
  EXPECT_EQ(both.intersection(nswOnly).getOverflowFlags(),
            OverflowFlags::Nsw | OverflowFlags::Nuw);
  ConstantIntRanges none(zero, two, zero, two, OverflowFlags::None);
  EXPECT_EQ(nswOnly.intersection(none).getOverflowFlags(), OverflowFlags::Nsw);

  // Full equality tracks both bounds and overflow proofs.
  EXPECT_FALSE(nswOnly == nswOnly.withOverflowFlags(OverflowFlags::None));
  // Bounds-only equality intentionally ignores overflow proofs.
  EXPECT_TRUE(nswOnly.hasSameBounds(
      nswOnly.withOverflowFlags(OverflowFlags::Nsw | OverflowFlags::Nuw)));
}

TEST(IntRangeAttrs, OverflowFlagsPrinting) {
  APInt zero = APInt::getZero(64);
  APInt one = zero + 1;

  auto toString = [](const ConstantIntRanges &r) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << r;
    return buf;
  };

  ConstantIntRanges noFlags(zero, one, zero, one);
  EXPECT_EQ(toString(noFlags), "unsigned : [0, 1] signed : [0, 1]");

  ConstantIntRanges nsw(zero, one, zero, one, OverflowFlags::Nsw);
  EXPECT_EQ(toString(nsw), "unsigned : [0, 1] signed : [0, 1] overflow<nsw>");

  ConstantIntRanges nuw(zero, one, zero, one, OverflowFlags::Nuw);
  EXPECT_EQ(toString(nuw), "unsigned : [0, 1] signed : [0, 1] overflow<nuw>");

  ConstantIntRanges both(zero, one, zero, one,
                         OverflowFlags::Nsw | OverflowFlags::Nuw);
  EXPECT_EQ(toString(both),
            "unsigned : [0, 1] signed : [0, 1] overflow<nsw, nuw>");
}

TEST(IntRangeAttrs, InferIndexOpCmpBothIgnoresOverflowFlags) {
  intrange::InferRangeFn inferFn = [](ArrayRef<ConstantIntRanges> args) {
    unsigned width = args.front().umin().getBitWidth();
    APInt zero = APInt::getZero(width);
    APInt one(width, 1);
    return ConstantIntRanges(zero, one, zero, one, OverflowFlags::Nsw);
  };

  APInt zero64 = APInt::getZero(64);
  APInt one64(64, 1);
  ConstantIntRanges arg(zero64, one64, zero64, one64);
  ConstantIntRanges result =
      intrange::inferIndexOp(inferFn, {arg}, intrange::CmpMode::Both);

  EXPECT_EQ(result.umin(), zero64);
  EXPECT_EQ(result.umax(), one64);
  EXPECT_EQ(result.smin(), zero64);
  EXPECT_EQ(result.smax(), one64);
  EXPECT_EQ(result.getOverflowFlags(), OverflowFlags::Nsw);
}

TEST(IntRangeAttrs, ArithAddIOpInfersOverflowFlags) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect>();

  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  ModuleOp module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 8);
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 8);

  arith::AddIOp add = arith::AddIOp::create(builder, loc, zero, one);
  arith::AddIOp addWithDeclaredNsw = arith::AddIOp::create(
      builder, loc, zero, one, arith::IntegerOverflowFlags::nsw);

  APInt c0(8, 0), c1(8, 1), c2(8, 2), c10(8, 10), c20(8, 20), c120(8, 120),
      c127(8, 127), c255(8, 255);

  // Both signed and unsigned proofs succeed.
  ConstantIntRanges addLhs(c0, c10, c0, c10);
  ConstantIntRanges addRhs(c0, c20, c0, c20);
  ConstantIntRanges addResult = inferBinaryOpResult(add, {addLhs, addRhs});
  EXPECT_EQ(addResult.getOverflowFlags(),
            OverflowFlags::Nsw | OverflowFlags::Nuw);

  // Signed may overflow, but unsigned remains provably no-wrap.
  ConstantIntRanges mayOverflowLhs(c120, c127, c120, c127);
  ConstantIntRanges mayOverflowRhs(c1, c2, c1, c2);
  ConstantIntRanges addMayOverflowResult =
      inferBinaryOpResult(add, {mayOverflowLhs, mayOverflowRhs});
  EXPECT_EQ(addMayOverflowResult.getOverflowFlags(), OverflowFlags::Nuw);

  // Declared op flags are preserved and merged with inferred ones.
  ConstantIntRanges addDeclaredNswResult =
      inferBinaryOpResult(addWithDeclaredNsw, {mayOverflowLhs, mayOverflowRhs});
  EXPECT_EQ(addDeclaredNswResult.getOverflowFlags(),
            OverflowFlags::Nsw | OverflowFlags::Nuw);

  // Both signed and unsigned proofs fail.
  ConstantIntRanges fullUnsigned = ConstantIntRanges::fromUnsigned(c0, c255);
  ConstantIntRanges addFullyOverflowingResult =
      inferBinaryOpResult(add, {fullUnsigned, fullUnsigned});
  EXPECT_EQ(addFullyOverflowingResult.getOverflowFlags(), OverflowFlags::None);
}

TEST(IntRangeAttrs, ArithSubIOpInfersOverflowFlags) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect>();

  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  ModuleOp module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 8);
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 8);

  arith::SubIOp sub = arith::SubIOp::create(builder, loc, zero, one);
  arith::SubIOp subWithDeclaredNuw = arith::SubIOp::create(
      builder, loc, zero, one, arith::IntegerOverflowFlags::nuw);

  APInt c0(8, 0), c5(8, 5), c10(8, 10), c20(8, 20);
  APInt sMin = APInt::getSignedMinValue(8);
  APInt sNeg120(8, -120, true);

  // Both signed and unsigned proofs succeed.
  ConstantIntRanges subLhs(c10, c20, c10, c20);
  ConstantIntRanges subRhs(c0, c5, c0, c5);
  ConstantIntRanges subResult = inferBinaryOpResult(sub, {subLhs, subRhs});
  EXPECT_EQ(subResult.getOverflowFlags(),
            OverflowFlags::Nsw | OverflowFlags::Nuw);

  // Signed may overflow, but unsigned remains provably no-wrap.
  ConstantIntRanges subMayOverflowLhs =
      ConstantIntRanges::fromSigned(sMin, sNeg120);
  ConstantIntRanges subMayOverflowRhs(c10, c20, c10, c20);
  ConstantIntRanges subMayOverflowResult =
      inferBinaryOpResult(sub, {subMayOverflowLhs, subMayOverflowRhs});
  EXPECT_EQ(subMayOverflowResult.getOverflowFlags(), OverflowFlags::Nuw);

  // Declared op flags are preserved.
  ConstantIntRanges subDeclaredNuwResult = inferBinaryOpResult(
      subWithDeclaredNuw, {subMayOverflowLhs, subMayOverflowRhs});
  EXPECT_EQ(subDeclaredNuwResult.getOverflowFlags(), OverflowFlags::Nuw);
}

TEST(IntRangeAttrs, ArithMulIOpInfersOverflowFlags) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect>();

  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  ModuleOp module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 8);
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 8);

  arith::MulIOp mul = arith::MulIOp::create(builder, loc, zero, one);
  arith::MulIOp mulWithDeclaredFlags = arith::MulIOp::create(
      builder, loc, zero, one,
      arith::IntegerOverflowFlags::nsw | arith::IntegerOverflowFlags::nuw);

  APInt c2(8, 2), c3(8, 3), c4(8, 4), c5(8, 5), c8(8, 8), c16(8, 16),
      c20(8, 20);

  // Both signed and unsigned proofs succeed.
  ConstantIntRanges mulLhs(c2, c3, c2, c3);
  ConstantIntRanges mulRhs(c4, c5, c4, c5);
  ConstantIntRanges mulResult = inferBinaryOpResult(mul, {mulLhs, mulRhs});
  EXPECT_EQ(mulResult.getOverflowFlags(),
            OverflowFlags::Nsw | OverflowFlags::Nuw);

  // Unsigned proof succeeds, but signed proof fails (16 * 8 = 128 in i8).
  ConstantIntRanges mulNuwOnlyLhs(c16, c16, c16, c16);
  ConstantIntRanges mulNuwOnlyRhs(c8, c8, c8, c8);
  ConstantIntRanges mulNuwOnlyResult =
      inferBinaryOpResult(mul, {mulNuwOnlyLhs, mulNuwOnlyRhs});
  EXPECT_EQ(mulNuwOnlyResult.getOverflowFlags(), OverflowFlags::Nuw);

  // Both signed and unsigned proofs fail.
  ConstantIntRanges mulMayOverflowLhs(c16, c20, c16, c20);
  ConstantIntRanges mulMayOverflowRhs(c16, c20, c16, c20);
  ConstantIntRanges mulMayOverflowResult =
      inferBinaryOpResult(mul, {mulMayOverflowLhs, mulMayOverflowRhs});
  EXPECT_EQ(mulMayOverflowResult.getOverflowFlags(), OverflowFlags::None);

  // Declared op flags are preserved.
  ConstantIntRanges mulDeclaredResult = inferBinaryOpResult(
      mulWithDeclaredFlags, {mulMayOverflowLhs, mulMayOverflowRhs});
  EXPECT_EQ(mulDeclaredResult.getOverflowFlags(),
            OverflowFlags::Nsw | OverflowFlags::Nuw);
}
