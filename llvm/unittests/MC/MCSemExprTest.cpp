//===- llvm/unittests/MC/MCSemExprTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSemExpr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Arbitrary physical register ids
const MCRegister R1(51);
const MCRegister R2(9);

TEST(MCSemExprTest, ConstantIsCanonical) {
  MCSemExpr E = MCSemExpr::createConst(5);
  EXPECT_TRUE(E.isConstant());
  EXPECT_EQ(E.getScale(), 0);
  EXPECT_EQ(E.getOffset(), 5);
  // A == 0 implies the invalid leaf reg, canonical for constant uniqueness.
  ASSERT_TRUE(E.getLeaf().isReg());
  EXPECT_FALSE(E.getLeaf().getReg().isValid());
  EXPECT_EQ(E, MCSemExpr::createConst(5));

  MCSemAddrExpr A = MCSemAddrExpr::createConst(0x1000);
  EXPECT_TRUE(A.isConstant());
  EXPECT_FALSE(A.getReg().isValid());
  EXPECT_EQ(A, MCSemAddrExpr::createConst(0x1000));
}

TEST(MCSemExprTest, Equality) {
  MCSemExpr E = MCSemExpr::createReg(1, R1, -16);
  EXPECT_EQ(E, MCSemExpr::createReg(1, R1, -16));
  EXPECT_NE(E, MCSemExpr::createReg(2, R1, -16));
  EXPECT_NE(E, MCSemExpr::createReg(1, R2, -16));
  EXPECT_NE(E, MCSemExpr::createReg(1, R1, 8));
  EXPECT_NE(E, MCSemExpr::createConst(-8));
  // Reading a register should be different from reading the memory it points
  // at.
  EXPECT_NE(E,
            MCSemExpr::createMem(1, MCSemAddrExpr::createReg(1, R1, -16), 0));
}

TEST(MCSemLeafTest, KindsAreNeverEqual) {
  // Leaf kind compared first, to prevent equality of inactive field in class.
  EXPECT_NE(MCSemLeaf::createReg(MCRegister()),
            MCSemLeaf::createMem(MCSemAddrExpr::createConst(0)));
}

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(MCSemExprTest, NonCanonicalDeath) {
  // A == 0 should be only used with createConst()
  EXPECT_DEATH((void)MCSemExpr::createReg(0, R1, 8), "non-canonical MCSemExpr");
  EXPECT_DEATH((void)MCSemAddrExpr::createReg(0, R1, 8),
               "non-canonical MCSemAddrExpr");
  // A register term requires a valid register.
  EXPECT_DEATH((void)MCSemExpr::createReg(1, MCRegister(), 8),
               "non-canonical MCSemExpr");
  EXPECT_DEATH((void)MCSemAddrExpr::createReg(1, MCRegister(), 8),
               "non-canonical MCSemAddrExpr");
  // Assert on the inactive leaf field.
  EXPECT_DEATH((void)MCSemLeaf::createReg(R1).getAddr(), "not a memory leaf");
  EXPECT_DEATH(
      (void)MCSemLeaf::createMem(MCSemAddrExpr::createConst(0)).getReg(),
      "not a register leaf");
}
#endif
#endif

} // namespace
