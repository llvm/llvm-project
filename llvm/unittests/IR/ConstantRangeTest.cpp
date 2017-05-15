//===- ConstantRangeTest.cpp - ConstantRange tests ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class ConstantRangeTest : public ::testing::Test {
protected:
  static ConstantRange Full;
  static ConstantRange Empty;
  static ConstantRange One;
  static ConstantRange Some;
  static ConstantRange Wrap;
};

ConstantRange ConstantRangeTest::Full(16);
ConstantRange ConstantRangeTest::Empty(16, false);
ConstantRange ConstantRangeTest::One(APInt(16, 0xa));
ConstantRange ConstantRangeTest::Some(APInt(16, 0xa), APInt(16, 0xaaa));
ConstantRange ConstantRangeTest::Wrap(APInt(16, 0xaaa), APInt(16, 0xa));

TEST_F(ConstantRangeTest, Basics) {
  EXPECT_TRUE(Full.isFullSet());
  EXPECT_FALSE(Full.isEmptySet());
  EXPECT_TRUE(Full.inverse().isEmptySet());
  EXPECT_FALSE(Full.isWrappedSet());
  EXPECT_TRUE(Full.contains(APInt(16, 0x0)));
  EXPECT_TRUE(Full.contains(APInt(16, 0x9)));
  EXPECT_TRUE(Full.contains(APInt(16, 0xa)));
  EXPECT_TRUE(Full.contains(APInt(16, 0xaa9)));
  EXPECT_TRUE(Full.contains(APInt(16, 0xaaa)));

  EXPECT_FALSE(Empty.isFullSet());
  EXPECT_TRUE(Empty.isEmptySet());
  EXPECT_TRUE(Empty.inverse().isFullSet());
  EXPECT_FALSE(Empty.isWrappedSet());
  EXPECT_FALSE(Empty.contains(APInt(16, 0x0)));
  EXPECT_FALSE(Empty.contains(APInt(16, 0x9)));
  EXPECT_FALSE(Empty.contains(APInt(16, 0xa)));
  EXPECT_FALSE(Empty.contains(APInt(16, 0xaa9)));
  EXPECT_FALSE(Empty.contains(APInt(16, 0xaaa)));

  EXPECT_FALSE(One.isFullSet());
  EXPECT_FALSE(One.isEmptySet());
  EXPECT_FALSE(One.isWrappedSet());
  EXPECT_FALSE(One.contains(APInt(16, 0x0)));
  EXPECT_FALSE(One.contains(APInt(16, 0x9)));
  EXPECT_TRUE(One.contains(APInt(16, 0xa)));
  EXPECT_FALSE(One.contains(APInt(16, 0xaa9)));
  EXPECT_FALSE(One.contains(APInt(16, 0xaaa)));
  EXPECT_FALSE(One.inverse().contains(APInt(16, 0xa)));

  EXPECT_FALSE(Some.isFullSet());
  EXPECT_FALSE(Some.isEmptySet());
  EXPECT_FALSE(Some.isWrappedSet());
  EXPECT_FALSE(Some.contains(APInt(16, 0x0)));
  EXPECT_FALSE(Some.contains(APInt(16, 0x9)));
  EXPECT_TRUE(Some.contains(APInt(16, 0xa)));
  EXPECT_TRUE(Some.contains(APInt(16, 0xaa9)));
  EXPECT_FALSE(Some.contains(APInt(16, 0xaaa)));

  EXPECT_FALSE(Wrap.isFullSet());
  EXPECT_FALSE(Wrap.isEmptySet());
  EXPECT_TRUE(Wrap.isWrappedSet());
  EXPECT_TRUE(Wrap.contains(APInt(16, 0x0)));
  EXPECT_TRUE(Wrap.contains(APInt(16, 0x9)));
  EXPECT_FALSE(Wrap.contains(APInt(16, 0xa)));
  EXPECT_FALSE(Wrap.contains(APInt(16, 0xaa9)));
  EXPECT_TRUE(Wrap.contains(APInt(16, 0xaaa)));
}

TEST_F(ConstantRangeTest, Equality) {
  EXPECT_EQ(Full, Full);
  EXPECT_EQ(Empty, Empty);
  EXPECT_EQ(One, One);
  EXPECT_EQ(Some, Some);
  EXPECT_EQ(Wrap, Wrap);
  EXPECT_NE(Full, Empty);
  EXPECT_NE(Full, One);
  EXPECT_NE(Full, Some);
  EXPECT_NE(Full, Wrap);
  EXPECT_NE(Empty, One);
  EXPECT_NE(Empty, Some);
  EXPECT_NE(Empty, Wrap);
  EXPECT_NE(One, Some);
  EXPECT_NE(One, Wrap);
  EXPECT_NE(Some, Wrap);
}

TEST_F(ConstantRangeTest, SingleElement) {
  EXPECT_EQ(Full.getSingleElement(), static_cast<APInt *>(nullptr));
  EXPECT_EQ(Empty.getSingleElement(), static_cast<APInt *>(nullptr));
  EXPECT_EQ(Full.getSingleMissingElement(), static_cast<APInt *>(nullptr));
  EXPECT_EQ(Empty.getSingleMissingElement(), static_cast<APInt *>(nullptr));

  EXPECT_EQ(*One.getSingleElement(), APInt(16, 0xa));
  EXPECT_EQ(Some.getSingleElement(), static_cast<APInt *>(nullptr));
  EXPECT_EQ(Wrap.getSingleElement(), static_cast<APInt *>(nullptr));

  EXPECT_EQ(One.getSingleMissingElement(), static_cast<APInt *>(nullptr));
  EXPECT_EQ(Some.getSingleMissingElement(), static_cast<APInt *>(nullptr));

  ConstantRange OneInverse = One.inverse();
  EXPECT_EQ(*OneInverse.getSingleMissingElement(), *One.getSingleElement());

  EXPECT_FALSE(Full.isSingleElement());
  EXPECT_FALSE(Empty.isSingleElement());
  EXPECT_TRUE(One.isSingleElement());
  EXPECT_FALSE(Some.isSingleElement());
  EXPECT_FALSE(Wrap.isSingleElement());
}

TEST_F(ConstantRangeTest, GetSetSize) {
  EXPECT_EQ(Full.getSetSize(), APInt(17, 65536));
  EXPECT_EQ(Empty.getSetSize(), APInt(17, 0));
  EXPECT_EQ(One.getSetSize(), APInt(17, 1));
  EXPECT_EQ(Some.getSetSize(), APInt(17, 0xaa0));

  ConstantRange Wrap(APInt(4, 7), APInt(4, 3));
  ConstantRange Wrap2(APInt(4, 8), APInt(4, 7));
  EXPECT_EQ(Wrap.getSetSize(), APInt(5, 12));
  EXPECT_EQ(Wrap2.getSetSize(), APInt(5, 15));
}

TEST_F(ConstantRangeTest, GetMinsAndMaxes) {
  EXPECT_EQ(Full.getUnsignedMax(), APInt(16, UINT16_MAX));
  EXPECT_EQ(One.getUnsignedMax(), APInt(16, 0xa));
  EXPECT_EQ(Some.getUnsignedMax(), APInt(16, 0xaa9));
  EXPECT_EQ(Wrap.getUnsignedMax(), APInt(16, UINT16_MAX));

  EXPECT_EQ(Full.getUnsignedMin(), APInt(16, 0));
  EXPECT_EQ(One.getUnsignedMin(), APInt(16, 0xa));
  EXPECT_EQ(Some.getUnsignedMin(), APInt(16, 0xa));
  EXPECT_EQ(Wrap.getUnsignedMin(), APInt(16, 0));

  EXPECT_EQ(Full.getSignedMax(), APInt(16, INT16_MAX));
  EXPECT_EQ(One.getSignedMax(), APInt(16, 0xa));
  EXPECT_EQ(Some.getSignedMax(), APInt(16, 0xaa9));
  EXPECT_EQ(Wrap.getSignedMax(), APInt(16, INT16_MAX));

  EXPECT_EQ(Full.getSignedMin(), APInt(16, (uint64_t)INT16_MIN));
  EXPECT_EQ(One.getSignedMin(), APInt(16, 0xa));
  EXPECT_EQ(Some.getSignedMin(), APInt(16, 0xa));
  EXPECT_EQ(Wrap.getSignedMin(), APInt(16, (uint64_t)INT16_MIN));

  // Found by Klee
  EXPECT_EQ(ConstantRange(APInt(4, 7), APInt(4, 0)).getSignedMax(),
            APInt(4, 7));
}

TEST_F(ConstantRangeTest, SignWrapped) {
  EXPECT_TRUE(Full.isSignWrappedSet());
  EXPECT_FALSE(Empty.isSignWrappedSet());
  EXPECT_FALSE(One.isSignWrappedSet());
  EXPECT_FALSE(Some.isSignWrappedSet());
  EXPECT_TRUE(Wrap.isSignWrappedSet());

  EXPECT_FALSE(ConstantRange(APInt(8, 127), APInt(8, 128)).isSignWrappedSet());
  EXPECT_TRUE(ConstantRange(APInt(8, 127), APInt(8, 129)).isSignWrappedSet());
  EXPECT_FALSE(ConstantRange(APInt(8, 128), APInt(8, 129)).isSignWrappedSet());
  EXPECT_TRUE(ConstantRange(APInt(8, 10), APInt(8, 9)).isSignWrappedSet());
  EXPECT_TRUE(ConstantRange(APInt(8, 10), APInt(8, 250)).isSignWrappedSet());
  EXPECT_FALSE(ConstantRange(APInt(8, 250), APInt(8, 10)).isSignWrappedSet());
  EXPECT_FALSE(ConstantRange(APInt(8, 250), APInt(8, 251)).isSignWrappedSet());
}

TEST_F(ConstantRangeTest, Trunc) {
  ConstantRange TFull = Full.truncate(10);
  ConstantRange TEmpty = Empty.truncate(10);
  ConstantRange TOne = One.truncate(10);
  ConstantRange TSome = Some.truncate(10);
  ConstantRange TWrap = Wrap.truncate(10);
  EXPECT_TRUE(TFull.isFullSet());
  EXPECT_TRUE(TEmpty.isEmptySet());
  EXPECT_EQ(TOne, ConstantRange(One.getLower().trunc(10),
                                One.getUpper().trunc(10)));
  EXPECT_TRUE(TSome.isFullSet());
}

TEST_F(ConstantRangeTest, ZExt) {
  ConstantRange ZFull = Full.zeroExtend(20);
  ConstantRange ZEmpty = Empty.zeroExtend(20);
  ConstantRange ZOne = One.zeroExtend(20);
  ConstantRange ZSome = Some.zeroExtend(20);
  ConstantRange ZWrap = Wrap.zeroExtend(20);
  EXPECT_EQ(ZFull, ConstantRange(APInt(20, 0), APInt(20, 0x10000)));
  EXPECT_TRUE(ZEmpty.isEmptySet());
  EXPECT_EQ(ZOne, ConstantRange(One.getLower().zext(20),
                                One.getUpper().zext(20)));
  EXPECT_EQ(ZSome, ConstantRange(Some.getLower().zext(20),
                                 Some.getUpper().zext(20)));
  EXPECT_EQ(ZWrap, ConstantRange(APInt(20, 0), APInt(20, 0x10000)));

  // zext([5, 0), 3->7) = [5, 8)
  ConstantRange FiveZero(APInt(3, 5), APInt(3, 0));
  EXPECT_EQ(FiveZero.zeroExtend(7), ConstantRange(APInt(7, 5), APInt(7, 8)));
}

TEST_F(ConstantRangeTest, SExt) {
  ConstantRange SFull = Full.signExtend(20);
  ConstantRange SEmpty = Empty.signExtend(20);
  ConstantRange SOne = One.signExtend(20);
  ConstantRange SSome = Some.signExtend(20);
  ConstantRange SWrap = Wrap.signExtend(20);
  EXPECT_EQ(SFull, ConstantRange(APInt(20, (uint64_t)INT16_MIN, true),
                                 APInt(20, INT16_MAX + 1, true)));
  EXPECT_TRUE(SEmpty.isEmptySet());
  EXPECT_EQ(SOne, ConstantRange(One.getLower().sext(20),
                                One.getUpper().sext(20)));
  EXPECT_EQ(SSome, ConstantRange(Some.getLower().sext(20),
                                 Some.getUpper().sext(20)));
  EXPECT_EQ(SWrap, ConstantRange(APInt(20, (uint64_t)INT16_MIN, true),
                                 APInt(20, INT16_MAX + 1, true)));

  EXPECT_EQ(ConstantRange(APInt(8, 120), APInt(8, 140)).signExtend(16),
            ConstantRange(APInt(16, -128), APInt(16, 128)));

  EXPECT_EQ(ConstantRange(APInt(16, 0x0200), APInt(16, 0x8000)).signExtend(19),
            ConstantRange(APInt(19, 0x0200), APInt(19, 0x8000)));
}

TEST_F(ConstantRangeTest, IntersectWith) {
  EXPECT_EQ(Empty.intersectWith(Full), Empty);
  EXPECT_EQ(Empty.intersectWith(Empty), Empty);
  EXPECT_EQ(Empty.intersectWith(One), Empty);
  EXPECT_EQ(Empty.intersectWith(Some), Empty);
  EXPECT_EQ(Empty.intersectWith(Wrap), Empty);
  EXPECT_EQ(Full.intersectWith(Full), Full);
  EXPECT_EQ(Some.intersectWith(Some), Some);
  EXPECT_EQ(Some.intersectWith(One), One);
  EXPECT_EQ(Full.intersectWith(One), One);
  EXPECT_EQ(Full.intersectWith(Some), Some);
  EXPECT_EQ(Some.intersectWith(Wrap), Empty);
  EXPECT_EQ(One.intersectWith(Wrap), Empty);
  EXPECT_EQ(One.intersectWith(Wrap), Wrap.intersectWith(One));

  // Klee generated testcase from PR4545.
  // The intersection of i16 [4, 2) and [6, 5) is disjoint, looking like
  // 01..4.6789ABCDEF where the dots represent values not in the intersection.
  ConstantRange LHS(APInt(16, 4), APInt(16, 2));
  ConstantRange RHS(APInt(16, 6), APInt(16, 5));
  EXPECT_TRUE(LHS.intersectWith(RHS) == LHS);

  // previous bug: intersection of [min, 3) and [2, max) should be 2
  LHS = ConstantRange(APInt(32, -2147483646), APInt(32, 3));
  RHS = ConstantRange(APInt(32, 2), APInt(32, 2147483646));
  EXPECT_EQ(LHS.intersectWith(RHS), ConstantRange(APInt(32, 2)));

  // [2, 0) /\ [4, 3) = [2, 0)
  LHS = ConstantRange(APInt(32, 2), APInt(32, 0));
  RHS = ConstantRange(APInt(32, 4), APInt(32, 3));
  EXPECT_EQ(LHS.intersectWith(RHS), ConstantRange(APInt(32, 2), APInt(32, 0)));

  // [2, 0) /\ [4, 2) = [4, 0)
  LHS = ConstantRange(APInt(32, 2), APInt(32, 0));
  RHS = ConstantRange(APInt(32, 4), APInt(32, 2));
  EXPECT_EQ(LHS.intersectWith(RHS), ConstantRange(APInt(32, 4), APInt(32, 0)));

  // [4, 2) /\ [5, 1) = [5, 1)
  LHS = ConstantRange(APInt(32, 4), APInt(32, 2));
  RHS = ConstantRange(APInt(32, 5), APInt(32, 1));
  EXPECT_EQ(LHS.intersectWith(RHS), ConstantRange(APInt(32, 5), APInt(32, 1)));

  // [2, 0) /\ [7, 4) = [7, 4)
  LHS = ConstantRange(APInt(32, 2), APInt(32, 0));
  RHS = ConstantRange(APInt(32, 7), APInt(32, 4));
  EXPECT_EQ(LHS.intersectWith(RHS), ConstantRange(APInt(32, 7), APInt(32, 4)));

  // [4, 2) /\ [1, 0) = [1, 0)
  LHS = ConstantRange(APInt(32, 4), APInt(32, 2));
  RHS = ConstantRange(APInt(32, 1), APInt(32, 0));
  EXPECT_EQ(LHS.intersectWith(RHS), ConstantRange(APInt(32, 4), APInt(32, 2)));
 
  // [15, 0) /\ [7, 6) = [15, 0)
  LHS = ConstantRange(APInt(32, 15), APInt(32, 0));
  RHS = ConstantRange(APInt(32, 7), APInt(32, 6));
  EXPECT_EQ(LHS.intersectWith(RHS), ConstantRange(APInt(32, 15), APInt(32, 0)));
}

TEST_F(ConstantRangeTest, UnionWith) {
  EXPECT_EQ(Wrap.unionWith(One),
            ConstantRange(APInt(16, 0xaaa), APInt(16, 0xb)));
  EXPECT_EQ(One.unionWith(Wrap), Wrap.unionWith(One));
  EXPECT_EQ(Empty.unionWith(Empty), Empty);
  EXPECT_EQ(Full.unionWith(Full), Full);
  EXPECT_EQ(Some.unionWith(Wrap), Full);

  // PR4545
  EXPECT_EQ(ConstantRange(APInt(16, 14), APInt(16, 1)).unionWith(
                                    ConstantRange(APInt(16, 0), APInt(16, 8))),
            ConstantRange(APInt(16, 14), APInt(16, 8)));
  EXPECT_EQ(ConstantRange(APInt(16, 6), APInt(16, 4)).unionWith(
                                    ConstantRange(APInt(16, 4), APInt(16, 0))),
              ConstantRange(16));
  EXPECT_EQ(ConstantRange(APInt(16, 1), APInt(16, 0)).unionWith(
                                    ConstantRange(APInt(16, 2), APInt(16, 1))),
              ConstantRange(16));
}

TEST_F(ConstantRangeTest, SetDifference) {
  EXPECT_EQ(Full.difference(Empty), Full);
  EXPECT_EQ(Full.difference(Full), Empty);
  EXPECT_EQ(Empty.difference(Empty), Empty);
  EXPECT_EQ(Empty.difference(Full), Empty);

  ConstantRange A(APInt(16, 3), APInt(16, 7));
  ConstantRange B(APInt(16, 5), APInt(16, 9));
  ConstantRange C(APInt(16, 3), APInt(16, 5));
  ConstantRange D(APInt(16, 7), APInt(16, 9));
  ConstantRange E(APInt(16, 5), APInt(16, 4));
  ConstantRange F(APInt(16, 7), APInt(16, 3));
  EXPECT_EQ(A.difference(B), C);
  EXPECT_EQ(B.difference(A), D);
  EXPECT_EQ(E.difference(A), F);
}

TEST_F(ConstantRangeTest, SubtractAPInt) {
  EXPECT_EQ(Full.subtract(APInt(16, 4)), Full);
  EXPECT_EQ(Empty.subtract(APInt(16, 4)), Empty);
  EXPECT_EQ(Some.subtract(APInt(16, 4)),
            ConstantRange(APInt(16, 0x6), APInt(16, 0xaa6)));
  EXPECT_EQ(Wrap.subtract(APInt(16, 4)),
            ConstantRange(APInt(16, 0xaa6), APInt(16, 0x6)));
  EXPECT_EQ(One.subtract(APInt(16, 4)),
            ConstantRange(APInt(16, 0x6)));
}

TEST_F(ConstantRangeTest, Add) {
  EXPECT_EQ(Full.add(APInt(16, 4)), Full);
  EXPECT_EQ(Full.add(Full), Full);
  EXPECT_EQ(Full.add(Empty), Empty);
  EXPECT_EQ(Full.add(One), Full);
  EXPECT_EQ(Full.add(Some), Full);
  EXPECT_EQ(Full.add(Wrap), Full);
  EXPECT_EQ(Empty.add(Empty), Empty);
  EXPECT_EQ(Empty.add(One), Empty);
  EXPECT_EQ(Empty.add(Some), Empty);
  EXPECT_EQ(Empty.add(Wrap), Empty);
  EXPECT_EQ(Empty.add(APInt(16, 4)), Empty);
  EXPECT_EQ(Some.add(APInt(16, 4)),
            ConstantRange(APInt(16, 0xe), APInt(16, 0xaae)));
  EXPECT_EQ(Wrap.add(APInt(16, 4)),
            ConstantRange(APInt(16, 0xaae), APInt(16, 0xe)));
  EXPECT_EQ(One.add(APInt(16, 4)),
            ConstantRange(APInt(16, 0xe)));
}

TEST_F(ConstantRangeTest, AddWithNoSignedWrap) {
  EXPECT_EQ(Empty.addWithNoSignedWrap(APInt(16, 1)), Empty);
  EXPECT_EQ(Full.addWithNoSignedWrap(APInt(16, 1)),
            ConstantRange(APInt(16, INT16_MIN+1), APInt(16, INT16_MIN)));
  EXPECT_EQ(ConstantRange(APInt(8, -50), APInt(8, 50)).addWithNoSignedWrap(APInt(8, 10)),
            ConstantRange(APInt(8, -40), APInt(8, 60)));
  EXPECT_EQ(ConstantRange(APInt(8, -50), APInt(8, 120)).addWithNoSignedWrap(APInt(8, 10)),
            ConstantRange(APInt(8, -40), APInt(8, INT8_MIN)));
  EXPECT_EQ(ConstantRange(APInt(8, 120), APInt(8, -10)).addWithNoSignedWrap(APInt(8, 5)),
            ConstantRange(APInt(8, 125), APInt(8, -5)));
  EXPECT_EQ(ConstantRange(APInt(8, 120), APInt(8, -120)).addWithNoSignedWrap(APInt(8, 10)),
            ConstantRange(APInt(8, INT8_MIN+10), APInt(8, -110)));

  EXPECT_EQ(Empty.addWithNoSignedWrap(APInt(16, -1)), Empty);
  EXPECT_EQ(Full.addWithNoSignedWrap(APInt(16, -1)),
            ConstantRange(APInt(16, INT16_MIN), APInt(16, INT16_MAX)));
  EXPECT_EQ(ConstantRange(APInt(8, -50), APInt(8, 50)).addWithNoSignedWrap(APInt(8, -10)),
            ConstantRange(APInt(8, -60), APInt(8, 40)));
  EXPECT_EQ(ConstantRange(APInt(8, -120), APInt(8, 50)).addWithNoSignedWrap(APInt(8, -10)),
            ConstantRange(APInt(8, INT8_MIN), APInt(8, 40)));
  EXPECT_EQ(ConstantRange(APInt(8, 120), APInt(8, -120)).addWithNoSignedWrap(APInt(8, -5)),
            ConstantRange(APInt(8, 115), APInt(8, -125)));
  EXPECT_EQ(ConstantRange(APInt(8, 120), APInt(8, -120)).addWithNoSignedWrap(APInt(8, -10)),
            ConstantRange(APInt(8, 110), APInt(8, INT8_MIN-10)));
}

TEST_F(ConstantRangeTest, Sub) {
  EXPECT_EQ(Full.sub(APInt(16, 4)), Full);
  EXPECT_EQ(Full.sub(Full), Full);
  EXPECT_EQ(Full.sub(Empty), Empty);
  EXPECT_EQ(Full.sub(One), Full);
  EXPECT_EQ(Full.sub(Some), Full);
  EXPECT_EQ(Full.sub(Wrap), Full);
  EXPECT_EQ(Empty.sub(Empty), Empty);
  EXPECT_EQ(Empty.sub(One), Empty);
  EXPECT_EQ(Empty.sub(Some), Empty);
  EXPECT_EQ(Empty.sub(Wrap), Empty);
  EXPECT_EQ(Empty.sub(APInt(16, 4)), Empty);
  EXPECT_EQ(Some.sub(APInt(16, 4)),
            ConstantRange(APInt(16, 0x6), APInt(16, 0xaa6)));
  EXPECT_EQ(Some.sub(Some),
            ConstantRange(APInt(16, 0xf561), APInt(16, 0xaa0)));
  EXPECT_EQ(Wrap.sub(APInt(16, 4)),
            ConstantRange(APInt(16, 0xaa6), APInt(16, 0x6)));
  EXPECT_EQ(One.sub(APInt(16, 4)),
            ConstantRange(APInt(16, 0x6)));
}

TEST_F(ConstantRangeTest, Multiply) {
  EXPECT_EQ(Full.multiply(Full), Full);
  EXPECT_EQ(Full.multiply(Empty), Empty);
  EXPECT_EQ(Full.multiply(One), Full);
  EXPECT_EQ(Full.multiply(Some), Full);
  EXPECT_EQ(Full.multiply(Wrap), Full);
  EXPECT_EQ(Empty.multiply(Empty), Empty);
  EXPECT_EQ(Empty.multiply(One), Empty);
  EXPECT_EQ(Empty.multiply(Some), Empty);
  EXPECT_EQ(Empty.multiply(Wrap), Empty);
  EXPECT_EQ(One.multiply(One), ConstantRange(APInt(16, 0xa*0xa),
                                             APInt(16, 0xa*0xa + 1)));
  EXPECT_EQ(One.multiply(Some), ConstantRange(APInt(16, 0xa*0xa),
                                              APInt(16, 0xa*0xaa9 + 1)));
  EXPECT_EQ(One.multiply(Wrap), Full);
  EXPECT_EQ(Some.multiply(Some), Full);
  EXPECT_EQ(Some.multiply(Wrap), Full);
  EXPECT_EQ(Wrap.multiply(Wrap), Full);

  ConstantRange Zero(APInt(16, 0));
  EXPECT_EQ(Zero.multiply(Full), Zero);
  EXPECT_EQ(Zero.multiply(Some), Zero);
  EXPECT_EQ(Zero.multiply(Wrap), Zero);
  EXPECT_EQ(Full.multiply(Zero), Zero);
  EXPECT_EQ(Some.multiply(Zero), Zero);
  EXPECT_EQ(Wrap.multiply(Zero), Zero);

  // http://llvm.org/PR4545
  EXPECT_EQ(ConstantRange(APInt(4, 1), APInt(4, 6)).multiply(
                ConstantRange(APInt(4, 6), APInt(4, 2))),
            ConstantRange(4, /*isFullSet=*/true));

  EXPECT_EQ(ConstantRange(APInt(8, 254), APInt(8, 0)).multiply(
              ConstantRange(APInt(8, 252), APInt(8, 4))),
            ConstantRange(APInt(8, 250), APInt(8, 9)));
  EXPECT_EQ(ConstantRange(APInt(8, 254), APInt(8, 255)).multiply(
              ConstantRange(APInt(8, 2), APInt(8, 4))),
            ConstantRange(APInt(8, 250), APInt(8, 253)));

  // TODO: This should be return [-2, 0]
  EXPECT_EQ(ConstantRange(APInt(8, -2)).multiply(
              ConstantRange(APInt(8, 0), APInt(8, 2))),
            ConstantRange(APInt(8, -2), APInt(8, 1)));
}

TEST_F(ConstantRangeTest, UMax) {
  EXPECT_EQ(Full.umax(Full), Full);
  EXPECT_EQ(Full.umax(Empty), Empty);
  EXPECT_EQ(Full.umax(Some), ConstantRange(APInt(16, 0xa), APInt(16, 0)));
  EXPECT_EQ(Full.umax(Wrap), Full);
  EXPECT_EQ(Full.umax(Some), ConstantRange(APInt(16, 0xa), APInt(16, 0)));
  EXPECT_EQ(Empty.umax(Empty), Empty);
  EXPECT_EQ(Empty.umax(Some), Empty);
  EXPECT_EQ(Empty.umax(Wrap), Empty);
  EXPECT_EQ(Empty.umax(One), Empty);
  EXPECT_EQ(Some.umax(Some), Some);
  EXPECT_EQ(Some.umax(Wrap), ConstantRange(APInt(16, 0xa), APInt(16, 0)));
  EXPECT_EQ(Some.umax(One), Some);
  // TODO: ConstantRange is currently over-conservative here.
  EXPECT_EQ(Wrap.umax(Wrap), Full);
  EXPECT_EQ(Wrap.umax(One), ConstantRange(APInt(16, 0xa), APInt(16, 0)));
  EXPECT_EQ(One.umax(One), One);
}

TEST_F(ConstantRangeTest, SMax) {
  EXPECT_EQ(Full.smax(Full), Full);
  EXPECT_EQ(Full.smax(Empty), Empty);
  EXPECT_EQ(Full.smax(Some), ConstantRange(APInt(16, 0xa),
                                           APInt::getSignedMinValue(16)));
  EXPECT_EQ(Full.smax(Wrap), Full);
  EXPECT_EQ(Full.smax(One), ConstantRange(APInt(16, 0xa),
                                          APInt::getSignedMinValue(16)));
  EXPECT_EQ(Empty.smax(Empty), Empty);
  EXPECT_EQ(Empty.smax(Some), Empty);
  EXPECT_EQ(Empty.smax(Wrap), Empty);
  EXPECT_EQ(Empty.smax(One), Empty);
  EXPECT_EQ(Some.smax(Some), Some);
  EXPECT_EQ(Some.smax(Wrap), ConstantRange(APInt(16, 0xa),
                                           APInt(16, (uint64_t)INT16_MIN)));
  EXPECT_EQ(Some.smax(One), Some);
  EXPECT_EQ(Wrap.smax(One), ConstantRange(APInt(16, 0xa),
                                          APInt(16, (uint64_t)INT16_MIN)));
  EXPECT_EQ(One.smax(One), One);
}

TEST_F(ConstantRangeTest, UMin) {
  EXPECT_EQ(Full.umin(Full), Full);
  EXPECT_EQ(Full.umin(Empty), Empty);
  EXPECT_EQ(Full.umin(Some), ConstantRange(APInt(16, 0), APInt(16, 0xaaa)));
  EXPECT_EQ(Full.umin(Wrap), Full);
  EXPECT_EQ(Empty.umin(Empty), Empty);
  EXPECT_EQ(Empty.umin(Some), Empty);
  EXPECT_EQ(Empty.umin(Wrap), Empty);
  EXPECT_EQ(Empty.umin(One), Empty);
  EXPECT_EQ(Some.umin(Some), Some);
  EXPECT_EQ(Some.umin(Wrap), ConstantRange(APInt(16, 0), APInt(16, 0xaaa)));
  EXPECT_EQ(Some.umin(One), One);
  // TODO: ConstantRange is currently over-conservative here.
  EXPECT_EQ(Wrap.umin(Wrap), Full);
  EXPECT_EQ(Wrap.umin(One), ConstantRange(APInt(16, 0), APInt(16, 0xb)));
  EXPECT_EQ(One.umin(One), One);
}

TEST_F(ConstantRangeTest, SMin) {
  EXPECT_EQ(Full.smin(Full), Full);
  EXPECT_EQ(Full.smin(Empty), Empty);
  EXPECT_EQ(Full.smin(Some), ConstantRange(APInt(16, (uint64_t)INT16_MIN),
                                           APInt(16, 0xaaa)));
  EXPECT_EQ(Full.smin(Wrap), Full);
  EXPECT_EQ(Empty.smin(Empty), Empty);
  EXPECT_EQ(Empty.smin(Some), Empty);
  EXPECT_EQ(Empty.smin(Wrap), Empty);
  EXPECT_EQ(Empty.smin(One), Empty);
  EXPECT_EQ(Some.smin(Some), Some);
  EXPECT_EQ(Some.smin(Wrap), ConstantRange(APInt(16, (uint64_t)INT16_MIN),
                                           APInt(16, 0xaaa)));
  EXPECT_EQ(Some.smin(One), One);
  // TODO: ConstantRange is currently over-conservative here.
  EXPECT_EQ(Wrap.smin(Wrap), Full);
  EXPECT_EQ(Wrap.smin(One), ConstantRange(APInt(16, (uint64_t)INT16_MIN),
                                          APInt(16, 0xb)));
  EXPECT_EQ(One.smin(One), One);
}

TEST_F(ConstantRangeTest, UDiv) {
  EXPECT_EQ(Full.udiv(Full), Full);
  EXPECT_EQ(Full.udiv(Empty), Empty);
  EXPECT_EQ(Full.udiv(One), ConstantRange(APInt(16, 0),
                                          APInt(16, 0xffff / 0xa + 1)));
  EXPECT_EQ(Full.udiv(Some), ConstantRange(APInt(16, 0),
                                           APInt(16, 0xffff / 0xa + 1)));
  EXPECT_EQ(Full.udiv(Wrap), Full);
  EXPECT_EQ(Empty.udiv(Empty), Empty);
  EXPECT_EQ(Empty.udiv(One), Empty);
  EXPECT_EQ(Empty.udiv(Some), Empty);
  EXPECT_EQ(Empty.udiv(Wrap), Empty);
  EXPECT_EQ(One.udiv(One), ConstantRange(APInt(16, 1)));
  EXPECT_EQ(One.udiv(Some), ConstantRange(APInt(16, 0), APInt(16, 2)));
  EXPECT_EQ(One.udiv(Wrap), ConstantRange(APInt(16, 0), APInt(16, 0xb)));
  EXPECT_EQ(Some.udiv(Some), ConstantRange(APInt(16, 0), APInt(16, 0x111)));
  EXPECT_EQ(Some.udiv(Wrap), ConstantRange(APInt(16, 0), APInt(16, 0xaaa)));
  EXPECT_EQ(Wrap.udiv(Wrap), Full);
}

TEST_F(ConstantRangeTest, Shl) {
  EXPECT_EQ(Full.shl(Full), Full);
  EXPECT_EQ(Full.shl(Empty), Empty);
  EXPECT_EQ(Full.shl(One), Full);    // TODO: [0, (-1 << 0xa) + 1)
  EXPECT_EQ(Full.shl(Some), Full);   // TODO: [0, (-1 << 0xa) + 1)
  EXPECT_EQ(Full.shl(Wrap), Full);
  EXPECT_EQ(Empty.shl(Empty), Empty);
  EXPECT_EQ(Empty.shl(One), Empty);
  EXPECT_EQ(Empty.shl(Some), Empty);
  EXPECT_EQ(Empty.shl(Wrap), Empty);
  EXPECT_EQ(One.shl(One), ConstantRange(APInt(16, 0xa << 0xa),
                                        APInt(16, (0xa << 0xa) + 1)));
  EXPECT_EQ(One.shl(Some), Full);    // TODO: [0xa << 0xa, 0)
  EXPECT_EQ(One.shl(Wrap), Full);    // TODO: [0xa, 0xa << 14 + 1)
  EXPECT_EQ(Some.shl(Some), Full);   // TODO: [0xa << 0xa, 0xfc01)
  EXPECT_EQ(Some.shl(Wrap), Full);   // TODO: [0xa, 0x7ff << 0x5 + 1)
  EXPECT_EQ(Wrap.shl(Wrap), Full);
}

TEST_F(ConstantRangeTest, Lshr) {
  EXPECT_EQ(Full.lshr(Full), Full);
  EXPECT_EQ(Full.lshr(Empty), Empty);
  EXPECT_EQ(Full.lshr(One), ConstantRange(APInt(16, 0),
                                          APInt(16, (0xffff >> 0xa) + 1)));
  EXPECT_EQ(Full.lshr(Some), ConstantRange(APInt(16, 0),
                                           APInt(16, (0xffff >> 0xa) + 1)));
  EXPECT_EQ(Full.lshr(Wrap), Full);
  EXPECT_EQ(Empty.lshr(Empty), Empty);
  EXPECT_EQ(Empty.lshr(One), Empty);
  EXPECT_EQ(Empty.lshr(Some), Empty);
  EXPECT_EQ(Empty.lshr(Wrap), Empty);
  EXPECT_EQ(One.lshr(One), ConstantRange(APInt(16, 0)));
  EXPECT_EQ(One.lshr(Some), ConstantRange(APInt(16, 0)));
  EXPECT_EQ(One.lshr(Wrap), ConstantRange(APInt(16, 0), APInt(16, 0xb)));
  EXPECT_EQ(Some.lshr(Some), ConstantRange(APInt(16, 0),
                                           APInt(16, (0xaaa >> 0xa) + 1)));
  EXPECT_EQ(Some.lshr(Wrap), ConstantRange(APInt(16, 0), APInt(16, 0xaaa)));
  EXPECT_EQ(Wrap.lshr(Wrap), Full);
}

TEST(ConstantRange, MakeAllowedICmpRegion) {
  // PR8250
  ConstantRange SMax = ConstantRange(APInt::getSignedMaxValue(32));
  EXPECT_TRUE(ConstantRange::makeAllowedICmpRegion(ICmpInst::ICMP_SGT, SMax)
                  .isEmptySet());
}

TEST(ConstantRange, MakeSatisfyingICmpRegion) {
  ConstantRange LowHalf(APInt(8, 0), APInt(8, 128));
  ConstantRange HighHalf(APInt(8, 128), APInt(8, 0));
  ConstantRange EmptySet(8, /* isFullSet = */ false);

  EXPECT_EQ(ConstantRange::makeSatisfyingICmpRegion(ICmpInst::ICMP_NE, LowHalf),
            HighHalf);

  EXPECT_EQ(
      ConstantRange::makeSatisfyingICmpRegion(ICmpInst::ICMP_NE, HighHalf),
      LowHalf);

  EXPECT_TRUE(ConstantRange::makeSatisfyingICmpRegion(ICmpInst::ICMP_EQ,
                                                      HighHalf).isEmptySet());

  ConstantRange UnsignedSample(APInt(8, 5), APInt(8, 200));

  EXPECT_EQ(ConstantRange::makeSatisfyingICmpRegion(ICmpInst::ICMP_ULT,
                                                    UnsignedSample),
            ConstantRange(APInt(8, 0), APInt(8, 5)));

  EXPECT_EQ(ConstantRange::makeSatisfyingICmpRegion(ICmpInst::ICMP_ULE,
                                                    UnsignedSample),
            ConstantRange(APInt(8, 0), APInt(8, 6)));

  EXPECT_EQ(ConstantRange::makeSatisfyingICmpRegion(ICmpInst::ICMP_UGT,
                                                    UnsignedSample),
            ConstantRange(APInt(8, 200), APInt(8, 0)));

  EXPECT_EQ(ConstantRange::makeSatisfyingICmpRegion(ICmpInst::ICMP_UGE,
                                                    UnsignedSample),
            ConstantRange(APInt(8, 199), APInt(8, 0)));

  ConstantRange SignedSample(APInt(8, -5), APInt(8, 5));

  EXPECT_EQ(
      ConstantRange::makeSatisfyingICmpRegion(ICmpInst::ICMP_SLT, SignedSample),
      ConstantRange(APInt(8, -128), APInt(8, -5)));

  EXPECT_EQ(
      ConstantRange::makeSatisfyingICmpRegion(ICmpInst::ICMP_SLE, SignedSample),
      ConstantRange(APInt(8, -128), APInt(8, -4)));

  EXPECT_EQ(
      ConstantRange::makeSatisfyingICmpRegion(ICmpInst::ICMP_SGT, SignedSample),
      ConstantRange(APInt(8, 5), APInt(8, -128)));

  EXPECT_EQ(
      ConstantRange::makeSatisfyingICmpRegion(ICmpInst::ICMP_SGE, SignedSample),
      ConstantRange(APInt(8, 4), APInt(8, -128)));
}

TEST(ConstantRange, MakeGuaranteedNoWrapRegion) {
  const int IntMin4Bits = 8;
  const int IntMax4Bits = 7;
  typedef OverflowingBinaryOperator OBO;

  for (int Const : {0, -1, -2, 1, 2, IntMin4Bits, IntMax4Bits}) {
    APInt C(4, Const, true /* = isSigned */);

    auto NUWRegion = ConstantRange::makeGuaranteedNoWrapRegion(
        Instruction::Add, C, OBO::NoUnsignedWrap);

    EXPECT_FALSE(NUWRegion.isEmptySet());

    auto NSWRegion = ConstantRange::makeGuaranteedNoWrapRegion(
        Instruction::Add, C, OBO::NoSignedWrap);

    EXPECT_FALSE(NSWRegion.isEmptySet());

    auto NoWrapRegion = ConstantRange::makeGuaranteedNoWrapRegion(
        Instruction::Add, C, OBO::NoSignedWrap | OBO::NoUnsignedWrap);

    EXPECT_FALSE(NoWrapRegion.isEmptySet());
    EXPECT_TRUE(NUWRegion.intersectWith(NSWRegion).contains(NoWrapRegion));

    for (APInt I = NUWRegion.getLower(), E = NUWRegion.getUpper(); I != E;
         ++I) {
      bool Overflow = false;
      (void)I.uadd_ov(C, Overflow);
      EXPECT_FALSE(Overflow);
    }

    for (APInt I = NSWRegion.getLower(), E = NSWRegion.getUpper(); I != E;
         ++I) {
      bool Overflow = false;
      (void)I.sadd_ov(C, Overflow);
      EXPECT_FALSE(Overflow);
    }

    for (APInt I = NoWrapRegion.getLower(), E = NoWrapRegion.getUpper(); I != E;
         ++I) {
      bool Overflow = false;

      (void)I.sadd_ov(C, Overflow);
      EXPECT_FALSE(Overflow);

      (void)I.uadd_ov(C, Overflow);
      EXPECT_FALSE(Overflow);
    }
  }

  auto NSWForAllValues = ConstantRange::makeGuaranteedNoWrapRegion(
      Instruction::Add, ConstantRange(32, /* isFullSet = */ true),
      OBO::NoSignedWrap);
  EXPECT_TRUE(NSWForAllValues.isSingleElement() &&
              NSWForAllValues.getSingleElement()->isMinValue());

  auto NUWForAllValues = ConstantRange::makeGuaranteedNoWrapRegion(
      Instruction::Add, ConstantRange(32, /* isFullSet = */ true),
      OBO::NoUnsignedWrap);
  EXPECT_TRUE(NUWForAllValues.isSingleElement() &&
              NUWForAllValues.getSingleElement()->isMinValue());

  auto NUWAndNSWForAllValues = ConstantRange::makeGuaranteedNoWrapRegion(
      Instruction::Add, ConstantRange(32, /* isFullSet = */ true),
      OBO::NoUnsignedWrap | OBO::NoSignedWrap);
  EXPECT_TRUE(NUWAndNSWForAllValues.isSingleElement() &&
              NUWAndNSWForAllValues.getSingleElement()->isMinValue());

  ConstantRange OneToFive(APInt(32, 1), APInt(32, 6));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, OneToFive, OBO::NoSignedWrap),
            ConstantRange(APInt::getSignedMinValue(32),
                          APInt::getSignedMaxValue(32) - 4));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, OneToFive, OBO::NoUnsignedWrap),
            ConstantRange(APInt::getMinValue(32), APInt::getMinValue(32) - 5));
  EXPECT_EQ(
      ConstantRange::makeGuaranteedNoWrapRegion(
          Instruction::Add, OneToFive, OBO::NoUnsignedWrap | OBO::NoSignedWrap),
      ConstantRange(APInt::getMinValue(32), APInt::getSignedMaxValue(32) - 4));

  ConstantRange MinusFiveToMinusTwo(APInt(32, -5), APInt(32, -1));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, MinusFiveToMinusTwo, OBO::NoSignedWrap),
            ConstantRange(APInt::getSignedMinValue(32) + 5,
                          APInt::getSignedMinValue(32)));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, MinusFiveToMinusTwo, OBO::NoUnsignedWrap),
            ConstantRange(APInt(32, 0), APInt(32, 2)));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, MinusFiveToMinusTwo,
                OBO::NoUnsignedWrap | OBO::NoSignedWrap),
            ConstantRange(APInt(32, 0), APInt(32, 2)));

  ConstantRange MinusOneToOne(APInt(32, -1), APInt(32, 2));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, MinusOneToOne, OBO::NoSignedWrap),
            ConstantRange(APInt::getSignedMinValue(32) + 1,
                          APInt::getSignedMinValue(32) - 1));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, MinusOneToOne, OBO::NoUnsignedWrap),
            ConstantRange(APInt(32, 0), APInt(32, 1)));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, MinusOneToOne,
                OBO::NoUnsignedWrap | OBO::NoSignedWrap),
            ConstantRange(APInt(32, 0), APInt(32, 1)));
}

TEST(ConstantRange, GetEquivalentICmp) {
  APInt RHS;
  CmpInst::Predicate Pred;

  EXPECT_TRUE(ConstantRange(APInt::getMinValue(32), APInt(32, 100))
                  .getEquivalentICmp(Pred, RHS));
  EXPECT_EQ(Pred, CmpInst::ICMP_ULT);
  EXPECT_EQ(RHS, APInt(32, 100));

  EXPECT_TRUE(ConstantRange(APInt::getSignedMinValue(32), APInt(32, 100))
                  .getEquivalentICmp(Pred, RHS));
  EXPECT_EQ(Pred, CmpInst::ICMP_SLT);
  EXPECT_EQ(RHS, APInt(32, 100));

  EXPECT_TRUE(ConstantRange(APInt(32, 100), APInt::getMinValue(32))
                  .getEquivalentICmp(Pred, RHS));
  EXPECT_EQ(Pred, CmpInst::ICMP_UGE);
  EXPECT_EQ(RHS, APInt(32, 100));

  EXPECT_TRUE(ConstantRange(APInt(32, 100), APInt::getSignedMinValue(32))
                  .getEquivalentICmp(Pred, RHS));
  EXPECT_EQ(Pred, CmpInst::ICMP_SGE);
  EXPECT_EQ(RHS, APInt(32, 100));

  EXPECT_TRUE(
      ConstantRange(32, /*isFullSet=*/true).getEquivalentICmp(Pred, RHS));
  EXPECT_EQ(Pred, CmpInst::ICMP_UGE);
  EXPECT_EQ(RHS, APInt(32, 0));

  EXPECT_TRUE(
      ConstantRange(32, /*isFullSet=*/false).getEquivalentICmp(Pred, RHS));
  EXPECT_EQ(Pred, CmpInst::ICMP_ULT);
  EXPECT_EQ(RHS, APInt(32, 0));

  EXPECT_FALSE(ConstantRange(APInt(32, 100), APInt(32, 200))
                   .getEquivalentICmp(Pred, RHS));

  EXPECT_FALSE(ConstantRange(APInt::getSignedMinValue(32) - APInt(32, 100),
                             APInt::getSignedMinValue(32) + APInt(32, 100))
                   .getEquivalentICmp(Pred, RHS));

  EXPECT_FALSE(ConstantRange(APInt::getMinValue(32) - APInt(32, 100),
                             APInt::getMinValue(32) + APInt(32, 100))
                   .getEquivalentICmp(Pred, RHS));

  EXPECT_TRUE(ConstantRange(APInt(32, 100)).getEquivalentICmp(Pred, RHS));
  EXPECT_EQ(Pred, CmpInst::ICMP_EQ);
  EXPECT_EQ(RHS, APInt(32, 100));

  EXPECT_TRUE(
      ConstantRange(APInt(32, 100)).inverse().getEquivalentICmp(Pred, RHS));
  EXPECT_EQ(Pred, CmpInst::ICMP_NE);
  EXPECT_EQ(RHS, APInt(32, 100));

  EXPECT_TRUE(
      ConstantRange(APInt(512, 100)).inverse().getEquivalentICmp(Pred, RHS));
  EXPECT_EQ(Pred, CmpInst::ICMP_NE);
  EXPECT_EQ(RHS, APInt(512, 100));

  // NB!  It would be correct for the following four calls to getEquivalentICmp
  // to return ordered predicates like CmpInst::ICMP_ULT or CmpInst::ICMP_UGT.
  // However, that's not the case today.

  EXPECT_TRUE(ConstantRange(APInt(32, 0)).getEquivalentICmp(Pred, RHS));
  EXPECT_EQ(Pred, CmpInst::ICMP_EQ);
  EXPECT_EQ(RHS, APInt(32, 0));

  EXPECT_TRUE(
      ConstantRange(APInt(32, 0)).inverse().getEquivalentICmp(Pred, RHS));
  EXPECT_EQ(Pred, CmpInst::ICMP_NE);
  EXPECT_EQ(RHS, APInt(32, 0));

  EXPECT_TRUE(ConstantRange(APInt(32, -1)).getEquivalentICmp(Pred, RHS));
  EXPECT_EQ(Pred, CmpInst::ICMP_EQ);
  EXPECT_EQ(RHS, APInt(32, -1));

  EXPECT_TRUE(
      ConstantRange(APInt(32, -1)).inverse().getEquivalentICmp(Pred, RHS));
  EXPECT_EQ(Pred, CmpInst::ICMP_NE);
  EXPECT_EQ(RHS, APInt(32, -1));
}

}  // anonymous namespace
