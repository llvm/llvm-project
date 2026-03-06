//===- UBMatchersTest.cpp - Unit tests for UB matchers --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/UB/IR/UBMatchers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OwningOpRef.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {

class UBMatchersTest : public testing::Test {
public:
  UBMatchersTest() { ctx.getOrLoadDialect<ub::UBDialect>(); }

protected:
  MLIRContext ctx;
  OpBuilder b{&ctx};
};

TEST_F(UBMatchersTest, MatchPoisonOp) {
  auto loc = UnknownLoc::get(&ctx);
  OwningOpRef<ub::PoisonOp> poisonOp =
      ub::PoisonOp::create(b, loc, b.getF32Type());
  Value poisonVal = poisonOp->getResult();

  // Match via Operation*.
  EXPECT_TRUE(matchPattern(poisonOp->getOperation(), ub::m_Poison()));
  // Match via Value.
  EXPECT_TRUE(matchPattern(poisonVal, ub::m_Poison()));
}

TEST_F(UBMatchersTest, MatchPoisonAttr) {
  auto poisonAttr = ub::PoisonAttr::get(&ctx);
  // Match via Attribute.
  EXPECT_TRUE(matchPattern(poisonAttr, ub::m_Poison()));
}

TEST_F(UBMatchersTest, MatchVectorPoisonOp) {
  auto loc = UnknownLoc::get(&ctx);
  auto vectorType = VectorType::get({4}, b.getI32Type());
  OwningOpRef<ub::PoisonOp> poisonOp = ub::PoisonOp::create(b, loc, vectorType);

  EXPECT_TRUE(matchPattern(poisonOp->getOperation(), ub::m_Poison()));
}

TEST_F(UBMatchersTest, NonPoisonAttrDoesNotMatch) {
  IntegerAttr intAttr = b.getI32IntegerAttr(42);
  EXPECT_FALSE(matchPattern(intAttr, ub::m_Poison()));

  FloatAttr floatAttr = b.getF32FloatAttr(1.0f);
  EXPECT_FALSE(matchPattern(floatAttr, ub::m_Poison()));
}

TEST_F(UBMatchersTest, NonPoisonOpDoesNotMatch) {
  ctx.getOrLoadDialect<arith::ArithDialect>();
  auto loc = UnknownLoc::get(&ctx);
  OwningOpRef<arith::ConstantOp> constOp =
      arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(42));
  EXPECT_FALSE(matchPattern(constOp->getOperation(), ub::m_Poison()));
}

TEST_F(UBMatchersTest, NullAttrDoesNotMatch) {
  Attribute nullAttr;
  EXPECT_FALSE(matchPattern(nullAttr, ub::m_Poison()));
}

} // namespace
