//===- ConstantRangeTest.cpp - ConstantRange tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/KnownBits.h"
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

template<typename Fn>
static void EnumerateConstantRanges(unsigned Bits, Fn TestFn) {
  unsigned Max = 1 << Bits;
  for (unsigned Lo = 0; Lo < Max; Lo++) {
    for (unsigned Hi = 0; Hi < Max; Hi++) {
      // Enforce ConstantRange invariant.
      if (Lo == Hi && Lo != 0 && Lo != Max - 1)
        continue;

      ConstantRange CR(APInt(Bits, Lo), APInt(Bits, Hi));
      TestFn(CR);
    }
  }
}

template<typename Fn>
static void EnumerateTwoConstantRanges(unsigned Bits, Fn TestFn) {
  EnumerateConstantRanges(Bits, [&](const ConstantRange &CR1) {
    EnumerateConstantRanges(Bits, [&](const ConstantRange &CR2) {
      TestFn(CR1, CR2);
    });
  });
}

template<typename Fn>
static void ForeachNumInConstantRange(const ConstantRange &CR, Fn TestFn) {
  if (CR.isFullSet()) {
    for (APInt N = APInt::getNullValue(CR.getBitWidth());;) {
      TestFn(N);
      if (N == APInt::getAllOnesValue(CR.getBitWidth()))
        break;
      ++N;
    }
    return;
  }
  for (APInt N = CR.getLower(); N != CR.getUpper(); ++N)
    TestFn(N);
}

ConstantRange ConstantRangeTest::Full(16, true);
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
  EXPECT_FALSE(Full.isSignWrappedSet());
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

TEST_F(ConstantRangeTest, UpperWrapped) {
  // The behavior here is the same as for isWrappedSet() / isSignWrappedSet().
  EXPECT_FALSE(Full.isUpperWrapped());
  EXPECT_FALSE(Empty.isUpperWrapped());
  EXPECT_FALSE(One.isUpperWrapped());
  EXPECT_FALSE(Some.isUpperWrapped());
  EXPECT_TRUE(Wrap.isUpperWrapped());
  EXPECT_FALSE(Full.isUpperSignWrapped());
  EXPECT_FALSE(Empty.isUpperSignWrapped());
  EXPECT_FALSE(One.isUpperSignWrapped());
  EXPECT_FALSE(Some.isUpperSignWrapped());
  EXPECT_TRUE(Wrap.isUpperSignWrapped());

  // The behavior differs if Upper is the Min/SignedMin value.
  ConstantRange CR1(APInt(8, 42), APInt::getMinValue(8));
  EXPECT_FALSE(CR1.isWrappedSet());
  EXPECT_TRUE(CR1.isUpperWrapped());

  ConstantRange CR2(APInt(8, 42), APInt::getSignedMinValue(8));
  EXPECT_FALSE(CR2.isSignWrappedSet());
  EXPECT_TRUE(CR2.isUpperSignWrapped());
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
  EXPECT_TRUE(TWrap.isFullSet());

  // trunc([2, 5), 3->2) = [2, 1)
  ConstantRange TwoFive(APInt(3, 2), APInt(3, 5));
  EXPECT_EQ(TwoFive.truncate(2), ConstantRange(APInt(2, 2), APInt(2, 1)));

  // trunc([2, 6), 3->2) = full
  ConstantRange TwoSix(APInt(3, 2), APInt(3, 6));
  EXPECT_TRUE(TwoSix.truncate(2).isFullSet());

  // trunc([5, 7), 3->2) = [1, 3)
  ConstantRange FiveSeven(APInt(3, 5), APInt(3, 7));
  EXPECT_EQ(FiveSeven.truncate(2), ConstantRange(APInt(2, 1), APInt(2, 3)));

  // trunc([7, 1), 3->2) = [3, 1)
  ConstantRange SevenOne(APInt(3, 7), APInt(3, 1));
  EXPECT_EQ(SevenOne.truncate(2), ConstantRange(APInt(2, 3), APInt(2, 1)));
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

template<typename Fn1, typename Fn2>
void testBinarySetOperationExhaustive(Fn1 OpFn, Fn2 InResultFn) {
  unsigned Bits = 4;
  EnumerateTwoConstantRanges(Bits,
      [=](const ConstantRange &CR1, const ConstantRange &CR2) {
        // Collect up to three contiguous unsigned ranges. The HaveInterrupt
        // variables are used determine when we have to switch to the next
        // range because the previous one ended.
        APInt Lower1(Bits, 0), Upper1(Bits, 0);
        APInt Lower2(Bits, 0), Upper2(Bits, 0);
        APInt Lower3(Bits, 0), Upper3(Bits, 0);
        bool HaveRange1 = false, HaveInterrupt1 = false;
        bool HaveRange2 = false, HaveInterrupt2 = false;
        bool HaveRange3 = false, HaveInterrupt3 = false;

        APInt Num(Bits, 0);
        for (unsigned I = 0, Limit = 1 << Bits; I < Limit; ++I, ++Num) {
          if (!InResultFn(CR1, CR2, Num)) {
            if (HaveRange3)
              HaveInterrupt3 = true;
            else if (HaveRange2)
              HaveInterrupt2 = true;
            else if (HaveRange1)
              HaveInterrupt1 = true;
            continue;
          }

          if (HaveRange3) {
            Upper3 = Num;
          } else if (HaveInterrupt2) {
            HaveRange3 = true;
            Lower3 = Upper3 = Num;
          } else if (HaveRange2) {
            Upper2 = Num;
          } else if (HaveInterrupt1) {
            HaveRange2 = true;
            Lower2 = Upper2 = Num;
          } else if (HaveRange1) {
            Upper1 = Num;
          } else {
            HaveRange1 = true;
            Lower1 = Upper1 = Num;
          }
        }

        assert(!HaveInterrupt3 && "Should have at most three ranges");

        ConstantRange SmallestCR = OpFn(CR1, CR2, ConstantRange::Smallest);
        ConstantRange UnsignedCR = OpFn(CR1, CR2, ConstantRange::Unsigned);
        ConstantRange SignedCR = OpFn(CR1, CR2, ConstantRange::Signed);

        if (!HaveRange1) {
          EXPECT_TRUE(SmallestCR.isEmptySet());
          EXPECT_TRUE(UnsignedCR.isEmptySet());
          EXPECT_TRUE(SignedCR.isEmptySet());
          return;
        }

        if (!HaveRange2) {
          if (Lower1 == Upper1 + 1) {
            EXPECT_TRUE(SmallestCR.isFullSet());
            EXPECT_TRUE(UnsignedCR.isFullSet());
            EXPECT_TRUE(SignedCR.isFullSet());
          } else {
            ConstantRange Expected(Lower1, Upper1 + 1);
            EXPECT_EQ(Expected, SmallestCR);
            EXPECT_EQ(Expected, UnsignedCR);
            EXPECT_EQ(Expected, SignedCR);
          }
          return;
        }

        ConstantRange Variant1(Bits, /*full*/ true);
        ConstantRange Variant2(Bits, /*full*/ true);
        if (!HaveRange3) {
          // Compute the two possible ways to cover two disjoint ranges.
          if (Lower1 != Upper2 + 1)
            Variant1 = ConstantRange(Lower1, Upper2 + 1);
          if (Lower2 != Upper1 + 1)
            Variant2 = ConstantRange(Lower2, Upper1 + 1);
        } else {
          // If we have three ranges, the first and last one have to be adjacent
          // to the unsigned domain. It's better to think of this as having two
          // holes, and we can construct one range using each hole.
          assert(Lower1.isNullValue() && Upper3.isMaxValue());
          Variant1 = ConstantRange(Lower2, Upper1 + 1);
          Variant2 = ConstantRange(Lower3, Upper2 + 1);
        }

        // Smallest: Smaller set, then any set.
        if (Variant1.isSizeStrictlySmallerThan(Variant2))
          EXPECT_EQ(Variant1, SmallestCR);
        else if (Variant2.isSizeStrictlySmallerThan(Variant1))
          EXPECT_EQ(Variant2, SmallestCR);
        else
          EXPECT_TRUE(Variant1 == SmallestCR || Variant2 == SmallestCR);

        // Unsigned: Non-wrapped set, then smaller set, then any set.
        bool Variant1Full = Variant1.isFullSet() || Variant1.isWrappedSet();
        bool Variant2Full = Variant2.isFullSet() || Variant2.isWrappedSet();
        if (!Variant1Full && Variant2Full)
          EXPECT_EQ(Variant1, UnsignedCR);
        else if (Variant1Full && !Variant2Full)
          EXPECT_EQ(Variant2, UnsignedCR);
        else if (Variant1.isSizeStrictlySmallerThan(Variant2))
          EXPECT_EQ(Variant1, UnsignedCR);
        else if (Variant2.isSizeStrictlySmallerThan(Variant1))
          EXPECT_EQ(Variant2, UnsignedCR);
        else
          EXPECT_TRUE(Variant1 == UnsignedCR || Variant2 == UnsignedCR);

        // Signed: Signed non-wrapped set, then smaller set, then any set.
        Variant1Full = Variant1.isFullSet() || Variant1.isSignWrappedSet();
        Variant2Full = Variant2.isFullSet() || Variant2.isSignWrappedSet();
        if (!Variant1Full && Variant2Full)
          EXPECT_EQ(Variant1, SignedCR);
        else if (Variant1Full && !Variant2Full)
          EXPECT_EQ(Variant2, SignedCR);
        else if (Variant1.isSizeStrictlySmallerThan(Variant2))
          EXPECT_EQ(Variant1, SignedCR);
        else if (Variant2.isSizeStrictlySmallerThan(Variant1))
          EXPECT_EQ(Variant2, SignedCR);
        else
          EXPECT_TRUE(Variant1 == SignedCR || Variant2 == SignedCR);
      });
}

TEST_F(ConstantRangeTest, IntersectWithExhaustive) {
  testBinarySetOperationExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2,
         ConstantRange::PreferredRangeType Type) {
        return CR1.intersectWith(CR2, Type);
      },
      [](const ConstantRange &CR1, const ConstantRange &CR2, const APInt &N) {
        return CR1.contains(N) && CR2.contains(N);
      });
}

TEST_F(ConstantRangeTest, UnionWithExhaustive) {
  testBinarySetOperationExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2,
         ConstantRange::PreferredRangeType Type) {
        return CR1.unionWith(CR2, Type);
      },
      [](const ConstantRange &CR1, const ConstantRange &CR2, const APInt &N) {
        return CR1.contains(N) || CR2.contains(N);
      });
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
            ConstantRange::getFull(16));
  EXPECT_EQ(ConstantRange(APInt(16, 1), APInt(16, 0)).unionWith(
                                    ConstantRange(APInt(16, 2), APInt(16, 1))),
            ConstantRange::getFull(16));
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
  ConstantRange Some2(APInt(16, 0xfff), APInt(16, 0x8000));
  ConstantRange WrapNullMax(APInt(16, 0x1), APInt(16, 0x0));
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
  EXPECT_EQ(
      Some2.shl(ConstantRange(APInt(16, 0x1))),
      ConstantRange(APInt(16, 0xfff << 0x1), APInt(16, 0x7fff << 0x1) + 1));
  EXPECT_EQ(One.shl(WrapNullMax), Full);
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

TEST_F(ConstantRangeTest, Ashr) {
  EXPECT_EQ(Full.ashr(Full), Full);
  EXPECT_EQ(Full.ashr(Empty), Empty);
  EXPECT_EQ(Full.ashr(One), ConstantRange(APInt(16, 0xffe0),
                                          APInt(16, (0x7fff >> 0xa) + 1 )));
  ConstantRange Small(APInt(16, 0xa), APInt(16, 0xb));
  EXPECT_EQ(Full.ashr(Small), ConstantRange(APInt(16, 0xffe0),
                                           APInt(16, (0x7fff >> 0xa) + 1 )));
  EXPECT_EQ(Full.ashr(Some), ConstantRange(APInt(16, 0xffe0),
                                           APInt(16, (0x7fff >> 0xa) + 1 )));
  EXPECT_EQ(Full.ashr(Wrap), Full);
  EXPECT_EQ(Empty.ashr(Empty), Empty);
  EXPECT_EQ(Empty.ashr(One), Empty);
  EXPECT_EQ(Empty.ashr(Some), Empty);
  EXPECT_EQ(Empty.ashr(Wrap), Empty);
  EXPECT_EQ(One.ashr(One), ConstantRange(APInt(16, 0)));
  EXPECT_EQ(One.ashr(Some), ConstantRange(APInt(16, 0)));
  EXPECT_EQ(One.ashr(Wrap), ConstantRange(APInt(16, 0), APInt(16, 0xb)));
  EXPECT_EQ(Some.ashr(Some), ConstantRange(APInt(16, 0),
                                           APInt(16, (0xaaa >> 0xa) + 1)));
  EXPECT_EQ(Some.ashr(Wrap), ConstantRange(APInt(16, 0), APInt(16, 0xaaa)));
  EXPECT_EQ(Wrap.ashr(Wrap), Full);
  ConstantRange Neg(APInt(16, 0xf3f0, true), APInt(16, 0xf7f8, true));
  EXPECT_EQ(Neg.ashr(Small), ConstantRange(APInt(16, 0xfffc, true),
                                           APInt(16, 0xfffe, true)));
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
  }

  for (int Const : {0, -1, -2, 1, 2, IntMin4Bits, IntMax4Bits}) {
    APInt C(4, Const, true /* = isSigned */);

    auto NUWRegion = ConstantRange::makeGuaranteedNoWrapRegion(
        Instruction::Sub, C, OBO::NoUnsignedWrap);

    EXPECT_FALSE(NUWRegion.isEmptySet());

    auto NSWRegion = ConstantRange::makeGuaranteedNoWrapRegion(
        Instruction::Sub, C, OBO::NoSignedWrap);

    EXPECT_FALSE(NSWRegion.isEmptySet());

    for (APInt I = NUWRegion.getLower(), E = NUWRegion.getUpper(); I != E;
         ++I) {
      bool Overflow = false;
      (void)I.usub_ov(C, Overflow);
      EXPECT_FALSE(Overflow);
    }

    for (APInt I = NSWRegion.getLower(), E = NSWRegion.getUpper(); I != E;
         ++I) {
      bool Overflow = false;
      (void)I.ssub_ov(C, Overflow);
      EXPECT_FALSE(Overflow);
    }
  }

  auto NSWForAllValues = ConstantRange::makeGuaranteedNoWrapRegion(
      Instruction::Add, ConstantRange(32, /* isFullSet = */ true),
      OBO::NoSignedWrap);
  EXPECT_TRUE(NSWForAllValues.isSingleElement() &&
              NSWForAllValues.getSingleElement()->isMinValue());

  NSWForAllValues = ConstantRange::makeGuaranteedNoWrapRegion(
      Instruction::Sub, ConstantRange(32, /* isFullSet = */ true),
      OBO::NoSignedWrap);
  EXPECT_TRUE(NSWForAllValues.isSingleElement() &&
              NSWForAllValues.getSingleElement()->isMaxValue());

  auto NUWForAllValues = ConstantRange::makeGuaranteedNoWrapRegion(
      Instruction::Add, ConstantRange(32, /* isFullSet = */ true),
      OBO::NoUnsignedWrap);
  EXPECT_TRUE(NUWForAllValues.isSingleElement() &&
              NUWForAllValues.getSingleElement()->isMinValue());

  NUWForAllValues = ConstantRange::makeGuaranteedNoWrapRegion(
      Instruction::Sub, ConstantRange(32, /* isFullSet = */ true),
      OBO::NoUnsignedWrap);
  EXPECT_TRUE(NUWForAllValues.isSingleElement() &&
              NUWForAllValues.getSingleElement()->isMaxValue());

  EXPECT_TRUE(ConstantRange::makeGuaranteedNoWrapRegion(
      Instruction::Add, APInt(32, 0), OBO::NoUnsignedWrap).isFullSet());
  EXPECT_TRUE(ConstantRange::makeGuaranteedNoWrapRegion(
      Instruction::Add, APInt(32, 0), OBO::NoSignedWrap).isFullSet());
  EXPECT_TRUE(ConstantRange::makeGuaranteedNoWrapRegion(
      Instruction::Sub, APInt(32, 0), OBO::NoUnsignedWrap).isFullSet());
  EXPECT_TRUE(ConstantRange::makeGuaranteedNoWrapRegion(
      Instruction::Sub, APInt(32, 0), OBO::NoSignedWrap).isFullSet());

  ConstantRange OneToFive(APInt(32, 1), APInt(32, 6));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, OneToFive, OBO::NoSignedWrap),
            ConstantRange(APInt::getSignedMinValue(32),
                          APInt::getSignedMaxValue(32) - 4));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, OneToFive, OBO::NoUnsignedWrap),
            ConstantRange(APInt::getMinValue(32), APInt::getMinValue(32) - 5));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Sub, OneToFive, OBO::NoSignedWrap),
            ConstantRange(APInt::getSignedMinValue(32) + 5,
                          APInt::getSignedMinValue(32)));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Sub, OneToFive, OBO::NoUnsignedWrap),
            ConstantRange(APInt::getMinValue(32) + 5, APInt::getMinValue(32)));

  ConstantRange MinusFiveToMinusTwo(APInt(32, -5), APInt(32, -1));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, MinusFiveToMinusTwo, OBO::NoSignedWrap),
            ConstantRange(APInt::getSignedMinValue(32) + 5,
                          APInt::getSignedMinValue(32)));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, MinusFiveToMinusTwo, OBO::NoUnsignedWrap),
            ConstantRange(APInt(32, 0), APInt(32, 2)));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Sub, MinusFiveToMinusTwo, OBO::NoSignedWrap),
            ConstantRange(APInt::getSignedMinValue(32),
                          APInt::getSignedMaxValue(32) - 4));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Sub, MinusFiveToMinusTwo, OBO::NoUnsignedWrap),
            ConstantRange(APInt::getMaxValue(32) - 1,
                          APInt::getMinValue(32)));

  ConstantRange MinusOneToOne(APInt(32, -1), APInt(32, 2));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, MinusOneToOne, OBO::NoSignedWrap),
            ConstantRange(APInt::getSignedMinValue(32) + 1,
                          APInt::getSignedMinValue(32) - 1));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, MinusOneToOne, OBO::NoUnsignedWrap),
            ConstantRange(APInt(32, 0), APInt(32, 1)));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Sub, MinusOneToOne, OBO::NoSignedWrap),
            ConstantRange(APInt::getSignedMinValue(32) + 1,
                          APInt::getSignedMinValue(32) - 1));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Sub, MinusOneToOne, OBO::NoUnsignedWrap),
            ConstantRange(APInt::getMaxValue(32),
                          APInt::getMinValue(32)));

  ConstantRange One(APInt(32, 1), APInt(32, 2));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, One, OBO::NoSignedWrap),
            ConstantRange(APInt::getSignedMinValue(32),
                          APInt::getSignedMaxValue(32)));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Add, One, OBO::NoUnsignedWrap),
            ConstantRange(APInt::getMinValue(32), APInt::getMaxValue(32)));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Sub, One, OBO::NoSignedWrap),
            ConstantRange(APInt::getSignedMinValue(32) + 1,
                          APInt::getSignedMinValue(32)));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Sub, One, OBO::NoUnsignedWrap),
            ConstantRange(APInt::getMinValue(32) + 1, APInt::getMinValue(32)));
}

template<typename Fn>
void TestNoWrapRegionExhaustive(Instruction::BinaryOps BinOp,
                                unsigned NoWrapKind, Fn OverflowFn) {
  // When using 4 bits this test needs ~3s on a debug build.
  unsigned Bits = 3;
  EnumerateTwoConstantRanges(Bits,
      [&](const ConstantRange &CR1, const ConstantRange &CR2) {
        if (CR2.isEmptySet())
          return;

        ConstantRange NoWrap =
            ConstantRange::makeGuaranteedNoWrapRegion(BinOp, CR2, NoWrapKind);
        ForeachNumInConstantRange(CR1, [&](const APInt &N1) {
          bool NoOverflow = true;
          ForeachNumInConstantRange(CR2, [&](const APInt &N2) {
            if (OverflowFn(N1, N2))
              NoOverflow = false;
          });
          EXPECT_EQ(NoOverflow, NoWrap.contains(N1));
        });
      });
}

// Show that makeGuaranteedNoWrapRegion is precise if only one of
// NoUnsignedWrap or NoSignedWrap is used.
TEST(ConstantRange, NoWrapRegionExhaustive) {
  TestNoWrapRegionExhaustive(
      Instruction::Add, OverflowingBinaryOperator::NoUnsignedWrap,
      [](const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.uadd_ov(N2, Overflow);
        return Overflow;
      });
  TestNoWrapRegionExhaustive(
      Instruction::Add, OverflowingBinaryOperator::NoSignedWrap,
      [](const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.sadd_ov(N2, Overflow);
        return Overflow;
      });
  TestNoWrapRegionExhaustive(
      Instruction::Sub, OverflowingBinaryOperator::NoUnsignedWrap,
      [](const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.usub_ov(N2, Overflow);
        return Overflow;
      });
  TestNoWrapRegionExhaustive(
      Instruction::Sub, OverflowingBinaryOperator::NoSignedWrap,
      [](const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.ssub_ov(N2, Overflow);
        return Overflow;
      });
  TestNoWrapRegionExhaustive(
      Instruction::Mul, OverflowingBinaryOperator::NoUnsignedWrap,
      [](const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.umul_ov(N2, Overflow);
        return Overflow;
      });
  TestNoWrapRegionExhaustive(
      Instruction::Mul, OverflowingBinaryOperator::NoSignedWrap,
      [](const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.smul_ov(N2, Overflow);
        return Overflow;
      });
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

TEST(ConstantRange, MakeGuaranteedNoWrapRegionMulUnsignedSingleValue) {
  typedef OverflowingBinaryOperator OBO;

  for (uint64_t I = std::numeric_limits<uint8_t>::min();
       I <= std::numeric_limits<uint8_t>::max(); I++) {
    auto Range = ConstantRange::makeGuaranteedNoWrapRegion(
        Instruction::Mul, ConstantRange(APInt(8, I), APInt(8, I + 1)),
        OBO::NoUnsignedWrap);

    for (uint64_t V = std::numeric_limits<uint8_t>::min();
         V <= std::numeric_limits<uint8_t>::max(); V++) {
      bool Overflow;
      (void)APInt(8, I).umul_ov(APInt(8, V), Overflow);
      EXPECT_EQ(!Overflow, Range.contains(APInt(8, V)));
    }
  }
}

TEST(ConstantRange, MakeGuaranteedNoWrapRegionMulSignedSingleValue) {
  typedef OverflowingBinaryOperator OBO;

  for (int64_t I = std::numeric_limits<int8_t>::min();
       I <= std::numeric_limits<int8_t>::max(); I++) {
    auto Range = ConstantRange::makeGuaranteedNoWrapRegion(
        Instruction::Mul,
        ConstantRange(APInt(8, I, /*isSigned=*/true),
                      APInt(8, I + 1, /*isSigned=*/true)),
        OBO::NoSignedWrap);

    for (int64_t V = std::numeric_limits<int8_t>::min();
         V <= std::numeric_limits<int8_t>::max(); V++) {
      bool Overflow;
      (void)APInt(8, I, /*isSigned=*/true)
          .smul_ov(APInt(8, V, /*isSigned=*/true), Overflow);
      EXPECT_EQ(!Overflow, Range.contains(APInt(8, V, /*isSigned=*/true)));
    }
  }
}

TEST(ConstantRange, MakeGuaranteedNoWrapRegionMulUnsignedRange) {
  typedef OverflowingBinaryOperator OBO;

  for (uint64_t Lo = std::numeric_limits<uint8_t>::min();
       Lo <= std::numeric_limits<uint8_t>::max(); Lo++) {
    for (uint64_t Hi = Lo; Hi <= std::numeric_limits<uint8_t>::max(); Hi++) {
      EXPECT_EQ(
          ConstantRange::makeGuaranteedNoWrapRegion(
              Instruction::Mul, ConstantRange(APInt(8, Lo), APInt(8, Hi + 1)),
              OBO::NoUnsignedWrap),
          ConstantRange::makeGuaranteedNoWrapRegion(
              Instruction::Mul, ConstantRange(APInt(8, Hi), APInt(8, Hi + 1)),
              OBO::NoUnsignedWrap));
    }
  }
}

TEST(ConstantRange, MakeGuaranteedNoWrapRegionMulSignedRange) {
  typedef OverflowingBinaryOperator OBO;

  int Lo = -12, Hi = 16;
  auto Range = ConstantRange::makeGuaranteedNoWrapRegion(
      Instruction::Mul,
      ConstantRange(APInt(8, Lo, /*isSigned=*/true),
                    APInt(8, Hi + 1, /*isSigned=*/true)),
      OBO::NoSignedWrap);

  for (int64_t V = std::numeric_limits<int8_t>::min();
       V <= std::numeric_limits<int8_t>::max(); V++) {
    bool AnyOverflow = false;
    for (int64_t I = Lo; I <= Hi; I++) {
      bool Overflow;
      (void)APInt(8, I, /*isSigned=*/true)
          .smul_ov(APInt(8, V, /*isSigned=*/true), Overflow);
      AnyOverflow |= Overflow;
    }
    EXPECT_EQ(!AnyOverflow, Range.contains(APInt(8, V, /*isSigned=*/true)));
  }
}

#define EXPECT_MAY_OVERFLOW(op) \
  EXPECT_EQ(ConstantRange::OverflowResult::MayOverflow, (op))
#define EXPECT_ALWAYS_OVERFLOWS(op) \
  EXPECT_EQ(ConstantRange::OverflowResult::AlwaysOverflows, (op))
#define EXPECT_NEVER_OVERFLOWS(op) \
  EXPECT_EQ(ConstantRange::OverflowResult::NeverOverflows, (op))

TEST_F(ConstantRangeTest, UnsignedAddOverflow) {
  // Ill-defined - may overflow is a conservative result.
  EXPECT_MAY_OVERFLOW(Some.unsignedAddMayOverflow(Empty));
  EXPECT_MAY_OVERFLOW(Empty.unsignedAddMayOverflow(Some));

  // Never overflow despite one full/wrap set.
  ConstantRange Zero(APInt::getNullValue(16));
  EXPECT_NEVER_OVERFLOWS(Full.unsignedAddMayOverflow(Zero));
  EXPECT_NEVER_OVERFLOWS(Wrap.unsignedAddMayOverflow(Zero));
  EXPECT_NEVER_OVERFLOWS(Zero.unsignedAddMayOverflow(Full));
  EXPECT_NEVER_OVERFLOWS(Zero.unsignedAddMayOverflow(Wrap));

  // But usually full/wrap always may overflow.
  EXPECT_MAY_OVERFLOW(Full.unsignedAddMayOverflow(One));
  EXPECT_MAY_OVERFLOW(Wrap.unsignedAddMayOverflow(One));
  EXPECT_MAY_OVERFLOW(One.unsignedAddMayOverflow(Full));
  EXPECT_MAY_OVERFLOW(One.unsignedAddMayOverflow(Wrap));

  ConstantRange A(APInt(16, 0xfd00), APInt(16, 0xfe00));
  ConstantRange B1(APInt(16, 0x0100), APInt(16, 0x0201));
  ConstantRange B2(APInt(16, 0x0100), APInt(16, 0x0202));
  EXPECT_NEVER_OVERFLOWS(A.unsignedAddMayOverflow(B1));
  EXPECT_MAY_OVERFLOW(A.unsignedAddMayOverflow(B2));
  EXPECT_NEVER_OVERFLOWS(B1.unsignedAddMayOverflow(A));
  EXPECT_MAY_OVERFLOW(B2.unsignedAddMayOverflow(A));

  ConstantRange C1(APInt(16, 0x0299), APInt(16, 0x0400));
  ConstantRange C2(APInt(16, 0x0300), APInt(16, 0x0400));
  EXPECT_MAY_OVERFLOW(A.unsignedAddMayOverflow(C1));
  EXPECT_ALWAYS_OVERFLOWS(A.unsignedAddMayOverflow(C2));
  EXPECT_MAY_OVERFLOW(C1.unsignedAddMayOverflow(A));
  EXPECT_ALWAYS_OVERFLOWS(C2.unsignedAddMayOverflow(A));
}

TEST_F(ConstantRangeTest, UnsignedSubOverflow) {
  // Ill-defined - may overflow is a conservative result.
  EXPECT_MAY_OVERFLOW(Some.unsignedSubMayOverflow(Empty));
  EXPECT_MAY_OVERFLOW(Empty.unsignedSubMayOverflow(Some));

  // Never overflow despite one full/wrap set.
  ConstantRange Zero(APInt::getNullValue(16));
  ConstantRange Max(APInt::getAllOnesValue(16));
  EXPECT_NEVER_OVERFLOWS(Full.unsignedSubMayOverflow(Zero));
  EXPECT_NEVER_OVERFLOWS(Wrap.unsignedSubMayOverflow(Zero));
  EXPECT_NEVER_OVERFLOWS(Max.unsignedSubMayOverflow(Full));
  EXPECT_NEVER_OVERFLOWS(Max.unsignedSubMayOverflow(Wrap));

  // But usually full/wrap always may overflow.
  EXPECT_MAY_OVERFLOW(Full.unsignedSubMayOverflow(One));
  EXPECT_MAY_OVERFLOW(Wrap.unsignedSubMayOverflow(One));
  EXPECT_MAY_OVERFLOW(One.unsignedSubMayOverflow(Full));
  EXPECT_MAY_OVERFLOW(One.unsignedSubMayOverflow(Wrap));

  ConstantRange A(APInt(16, 0x0000), APInt(16, 0x0100));
  ConstantRange B(APInt(16, 0x0100), APInt(16, 0x0200));
  EXPECT_NEVER_OVERFLOWS(B.unsignedSubMayOverflow(A));
  EXPECT_ALWAYS_OVERFLOWS(A.unsignedSubMayOverflow(B));

  ConstantRange A1(APInt(16, 0x0000), APInt(16, 0x0101));
  ConstantRange B1(APInt(16, 0x0100), APInt(16, 0x0201));
  EXPECT_NEVER_OVERFLOWS(B1.unsignedSubMayOverflow(A1));
  EXPECT_MAY_OVERFLOW(A1.unsignedSubMayOverflow(B1));

  ConstantRange A2(APInt(16, 0x0000), APInt(16, 0x0102));
  ConstantRange B2(APInt(16, 0x0100), APInt(16, 0x0202));
  EXPECT_MAY_OVERFLOW(B2.unsignedSubMayOverflow(A2));
  EXPECT_MAY_OVERFLOW(A2.unsignedSubMayOverflow(B2));
}

TEST_F(ConstantRangeTest, SignedAddOverflow) {
  // Ill-defined - may overflow is a conservative result.
  EXPECT_MAY_OVERFLOW(Some.signedAddMayOverflow(Empty));
  EXPECT_MAY_OVERFLOW(Empty.signedAddMayOverflow(Some));

  // Never overflow despite one full/wrap set.
  ConstantRange Zero(APInt::getNullValue(16));
  EXPECT_NEVER_OVERFLOWS(Full.signedAddMayOverflow(Zero));
  EXPECT_NEVER_OVERFLOWS(Wrap.signedAddMayOverflow(Zero));
  EXPECT_NEVER_OVERFLOWS(Zero.signedAddMayOverflow(Full));
  EXPECT_NEVER_OVERFLOWS(Zero.signedAddMayOverflow(Wrap));

  // But usually full/wrap always may overflow.
  EXPECT_MAY_OVERFLOW(Full.signedAddMayOverflow(One));
  EXPECT_MAY_OVERFLOW(Wrap.signedAddMayOverflow(One));
  EXPECT_MAY_OVERFLOW(One.signedAddMayOverflow(Full));
  EXPECT_MAY_OVERFLOW(One.signedAddMayOverflow(Wrap));

  ConstantRange A(APInt(16, 0x7d00), APInt(16, 0x7e00));
  ConstantRange B1(APInt(16, 0x0100), APInt(16, 0x0201));
  ConstantRange B2(APInt(16, 0x0100), APInt(16, 0x0202));
  EXPECT_NEVER_OVERFLOWS(A.signedAddMayOverflow(B1));
  EXPECT_MAY_OVERFLOW(A.signedAddMayOverflow(B2));
  ConstantRange B3(APInt(16, 0x8000), APInt(16, 0x0201));
  ConstantRange B4(APInt(16, 0x8000), APInt(16, 0x0202));
  EXPECT_NEVER_OVERFLOWS(A.signedAddMayOverflow(B3));
  EXPECT_MAY_OVERFLOW(A.signedAddMayOverflow(B4));
  ConstantRange B5(APInt(16, 0x0299), APInt(16, 0x0400));
  ConstantRange B6(APInt(16, 0x0300), APInt(16, 0x0400));
  EXPECT_MAY_OVERFLOW(A.signedAddMayOverflow(B5));
  EXPECT_ALWAYS_OVERFLOWS(A.signedAddMayOverflow(B6));

  ConstantRange C(APInt(16, 0x8200), APInt(16, 0x8300));
  ConstantRange D1(APInt(16, 0xfe00), APInt(16, 0xff00));
  ConstantRange D2(APInt(16, 0xfd99), APInt(16, 0xff00));
  EXPECT_NEVER_OVERFLOWS(C.signedAddMayOverflow(D1));
  EXPECT_MAY_OVERFLOW(C.signedAddMayOverflow(D2));
  ConstantRange D3(APInt(16, 0xfe00), APInt(16, 0x8000));
  ConstantRange D4(APInt(16, 0xfd99), APInt(16, 0x8000));
  EXPECT_NEVER_OVERFLOWS(C.signedAddMayOverflow(D3));
  EXPECT_MAY_OVERFLOW(C.signedAddMayOverflow(D4));
  ConstantRange D5(APInt(16, 0xfc00), APInt(16, 0xfd02));
  ConstantRange D6(APInt(16, 0xfc00), APInt(16, 0xfd01));
  EXPECT_MAY_OVERFLOW(C.signedAddMayOverflow(D5));
  EXPECT_ALWAYS_OVERFLOWS(C.signedAddMayOverflow(D6));

  ConstantRange E(APInt(16, 0xff00), APInt(16, 0x0100));
  EXPECT_NEVER_OVERFLOWS(E.signedAddMayOverflow(E));
  ConstantRange F(APInt(16, 0xf000), APInt(16, 0x7000));
  EXPECT_MAY_OVERFLOW(F.signedAddMayOverflow(F));
}

TEST_F(ConstantRangeTest, SignedSubOverflow) {
  // Ill-defined - may overflow is a conservative result.
  EXPECT_MAY_OVERFLOW(Some.signedSubMayOverflow(Empty));
  EXPECT_MAY_OVERFLOW(Empty.signedSubMayOverflow(Some));

  // Never overflow despite one full/wrap set.
  ConstantRange Zero(APInt::getNullValue(16));
  EXPECT_NEVER_OVERFLOWS(Full.signedSubMayOverflow(Zero));
  EXPECT_NEVER_OVERFLOWS(Wrap.signedSubMayOverflow(Zero));

  // But usually full/wrap always may overflow.
  EXPECT_MAY_OVERFLOW(Full.signedSubMayOverflow(One));
  EXPECT_MAY_OVERFLOW(Wrap.signedSubMayOverflow(One));
  EXPECT_MAY_OVERFLOW(One.signedSubMayOverflow(Full));
  EXPECT_MAY_OVERFLOW(One.signedSubMayOverflow(Wrap));

  ConstantRange A(APInt(16, 0x7d00), APInt(16, 0x7e00));
  ConstantRange B1(APInt(16, 0xfe00), APInt(16, 0xff00));
  ConstantRange B2(APInt(16, 0xfd99), APInt(16, 0xff00));
  EXPECT_NEVER_OVERFLOWS(A.signedSubMayOverflow(B1));
  EXPECT_MAY_OVERFLOW(A.signedSubMayOverflow(B2));
  ConstantRange B3(APInt(16, 0xfc00), APInt(16, 0xfd02));
  ConstantRange B4(APInt(16, 0xfc00), APInt(16, 0xfd01));
  EXPECT_MAY_OVERFLOW(A.signedSubMayOverflow(B3));
  EXPECT_ALWAYS_OVERFLOWS(A.signedSubMayOverflow(B4));

  ConstantRange C(APInt(16, 0x8200), APInt(16, 0x8300));
  ConstantRange D1(APInt(16, 0x0100), APInt(16, 0x0201));
  ConstantRange D2(APInt(16, 0x0100), APInt(16, 0x0202));
  EXPECT_NEVER_OVERFLOWS(C.signedSubMayOverflow(D1));
  EXPECT_MAY_OVERFLOW(C.signedSubMayOverflow(D2));
  ConstantRange D3(APInt(16, 0x0299), APInt(16, 0x0400));
  ConstantRange D4(APInt(16, 0x0300), APInt(16, 0x0400));
  EXPECT_MAY_OVERFLOW(C.signedSubMayOverflow(D3));
  EXPECT_ALWAYS_OVERFLOWS(C.signedSubMayOverflow(D4));

  ConstantRange E(APInt(16, 0xff00), APInt(16, 0x0100));
  EXPECT_NEVER_OVERFLOWS(E.signedSubMayOverflow(E));
  ConstantRange F(APInt(16, 0xf000), APInt(16, 0x7001));
  EXPECT_MAY_OVERFLOW(F.signedSubMayOverflow(F));
}

template<typename Fn1, typename Fn2>
static void TestOverflowExhaustive(Fn1 OverflowFn, Fn2 MayOverflowFn) {
  // Constant range overflow checks are tested exhaustively on 4-bit numbers.
  unsigned Bits = 4;
  EnumerateTwoConstantRanges(Bits, [=](const ConstantRange &CR1,
                                       const ConstantRange &CR2) {
    unsigned Size1 = CR1.isFullSet()
                         ? 1u << CR1.getBitWidth()
                         : (CR1.getUpper() - CR1.getLower()).getZExtValue();
    unsigned Size2 = CR2.isFullSet()
                         ? 1u << CR2.getBitWidth()
                         : (CR2.getUpper() - CR2.getLower()).getZExtValue();

    // Loop over all N1 in CR1 and N2 in CR2 and check whether any of the
    // operations have overflow / have no overflow. These loops are based
    // on Size1/Size2 to properly handle empty/full ranges.
    bool RangeHasOverflow = false;
    bool RangeHasNoOverflow = false;
    APInt N1 = CR1.getLower();
    for (unsigned I1 = 0; I1 < Size1; ++I1, ++N1) {
      APInt N2 = CR2.getLower();
      for (unsigned I2 = 0; I2 < Size2; ++I2, ++N2) {
        assert(CR1.contains(N1));
        assert(CR2.contains(N2));

        if (OverflowFn(N1, N2))
          RangeHasOverflow = true;
        else
          RangeHasNoOverflow = true;
      }
    }

    ConstantRange::OverflowResult OR = MayOverflowFn(CR1, CR2);
    switch (OR) {
    case ConstantRange::OverflowResult::AlwaysOverflows:
      EXPECT_TRUE(RangeHasOverflow);
      EXPECT_FALSE(RangeHasNoOverflow);
      break;
    case ConstantRange::OverflowResult::NeverOverflows:
      EXPECT_FALSE(RangeHasOverflow);
      EXPECT_TRUE(RangeHasNoOverflow);
      break;
    case ConstantRange::OverflowResult::MayOverflow:
      // We return MayOverflow for empty sets as a conservative result,
      // but of course neither the RangeHasOverflow nor the
      // RangeHasNoOverflow flags will be set.
      if (CR1.isEmptySet() || CR2.isEmptySet())
        break;

      EXPECT_TRUE(RangeHasOverflow);
      EXPECT_TRUE(RangeHasNoOverflow);
      break;
    }
  });
}

TEST_F(ConstantRangeTest, UnsignedAddOverflowExhaustive) {
  TestOverflowExhaustive(
      [](const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.uadd_ov(N2, Overflow);
        return Overflow;
      },
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.unsignedAddMayOverflow(CR2);
      });
}

TEST_F(ConstantRangeTest, UnsignedSubOverflowExhaustive) {
  TestOverflowExhaustive(
      [](const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.usub_ov(N2, Overflow);
        return Overflow;
      },
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.unsignedSubMayOverflow(CR2);
      });
}

TEST_F(ConstantRangeTest, UnsignedMulOverflowExhaustive) {
  TestOverflowExhaustive(
      [](const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.umul_ov(N2, Overflow);
        return Overflow;
      },
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.unsignedMulMayOverflow(CR2);
      });
}

TEST_F(ConstantRangeTest, SignedAddOverflowExhaustive) {
  TestOverflowExhaustive(
      [](const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.sadd_ov(N2, Overflow);
        return Overflow;
      },
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.signedAddMayOverflow(CR2);
      });
}

TEST_F(ConstantRangeTest, SignedSubOverflowExhaustive) {
  TestOverflowExhaustive(
      [](const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.ssub_ov(N2, Overflow);
        return Overflow;
      },
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.signedSubMayOverflow(CR2);
      });
}

TEST_F(ConstantRangeTest, FromKnownBits) {
  KnownBits Unknown(16);
  EXPECT_EQ(Full, ConstantRange::fromKnownBits(Unknown, /*signed*/false));
  EXPECT_EQ(Full, ConstantRange::fromKnownBits(Unknown, /*signed*/true));

  // .10..01. -> unsigned 01000010 (66)  to 11011011 (219)
  //          -> signed   11000010 (194) to 01011011 (91)
  KnownBits Known(8);
  Known.Zero = 36;
  Known.One = 66;
  ConstantRange Unsigned(APInt(8, 66), APInt(8, 219 + 1));
  ConstantRange Signed(APInt(8, 194), APInt(8, 91 + 1));
  EXPECT_EQ(Unsigned, ConstantRange::fromKnownBits(Known, /*signed*/false));
  EXPECT_EQ(Signed, ConstantRange::fromKnownBits(Known, /*signed*/true));

  // 1.10.10. -> 10100100 (164) to 11101101 (237)
  Known.Zero = 18;
  Known.One = 164;
  ConstantRange CR1(APInt(8, 164), APInt(8, 237 + 1));
  EXPECT_EQ(CR1, ConstantRange::fromKnownBits(Known, /*signed*/false));
  EXPECT_EQ(CR1, ConstantRange::fromKnownBits(Known, /*signed*/true));

  // 01.0.1.0 -> 01000100 (68) to 01101110 (110)
  Known.Zero = 145;
  Known.One = 68;
  ConstantRange CR2(APInt(8, 68), APInt(8, 110 + 1));
  EXPECT_EQ(CR2, ConstantRange::fromKnownBits(Known, /*signed*/false));
  EXPECT_EQ(CR2, ConstantRange::fromKnownBits(Known, /*signed*/true));
}

TEST_F(ConstantRangeTest, FromKnownBitsExhaustive) {
  unsigned Bits = 4;
  unsigned Max = 1 << Bits;
  KnownBits Known(Bits);
  for (unsigned Zero = 0; Zero < Max; ++Zero) {
    for (unsigned One = 0; One < Max; ++One) {
      Known.Zero = Zero;
      Known.One = One;
      if (Known.hasConflict() || Known.isUnknown())
        continue;

      APInt MinUnsigned = APInt::getMaxValue(Bits);
      APInt MaxUnsigned = APInt::getMinValue(Bits);
      APInt MinSigned = APInt::getSignedMaxValue(Bits);
      APInt MaxSigned = APInt::getSignedMinValue(Bits);
      for (unsigned N = 0; N < Max; ++N) {
        APInt Num(Bits, N);
        if ((Num & Known.Zero) != 0 || (~Num & Known.One) != 0)
          continue;

        if (Num.ult(MinUnsigned)) MinUnsigned = Num;
        if (Num.ugt(MaxUnsigned)) MaxUnsigned = Num;
        if (Num.slt(MinSigned)) MinSigned = Num;
        if (Num.sgt(MaxSigned)) MaxSigned = Num;
      }

      ConstantRange UnsignedCR(MinUnsigned, MaxUnsigned + 1);
      ConstantRange SignedCR(MinSigned, MaxSigned + 1);
      EXPECT_EQ(UnsignedCR, ConstantRange::fromKnownBits(Known, false));
      EXPECT_EQ(SignedCR, ConstantRange::fromKnownBits(Known, true));
    }
  }
}

TEST_F(ConstantRangeTest, Negative) {
  // All elements in an empty set (of which there are none) are both negative
  // and non-negative. Empty & full sets checked explicitly for clarity, but
  // they are also covered by the exhaustive test below.
  EXPECT_TRUE(Empty.isAllNegative());
  EXPECT_TRUE(Empty.isAllNonNegative());
  EXPECT_FALSE(Full.isAllNegative());
  EXPECT_FALSE(Full.isAllNonNegative());

  unsigned Bits = 4;
  EnumerateConstantRanges(Bits, [](const ConstantRange &CR) {
    bool AllNegative = true;
    bool AllNonNegative = true;
    ForeachNumInConstantRange(CR, [&](const APInt &N) {
      if (!N.isNegative())
        AllNegative = false;
      if (!N.isNonNegative())
        AllNonNegative = false;
    });
    assert((CR.isEmptySet() || !AllNegative || !AllNonNegative) &&
           "Only empty set can be both all negative and all non-negative");

    EXPECT_EQ(AllNegative, CR.isAllNegative());
    EXPECT_EQ(AllNonNegative, CR.isAllNonNegative());
  });
}

}  // anonymous namespace
