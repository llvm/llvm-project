//===- ConstantRangeTest.cpp - ConstantRange tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ConstantRange.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
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
static void EnumerateAPInts(unsigned Bits, Fn TestFn) {
  APInt N(Bits, 0);
  do {
    TestFn(N);
  } while (++N != 0);
}

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

template <typename Fn>
static void EnumerateInterestingConstantRanges(Fn TestFn) {
  // Check 1 bit ranges, because they may have special cases.
  EnumerateConstantRanges(/* Bits */ 1, TestFn);
  // Check 4 bit ranges to have decent coverage without being too slow.
  EnumerateConstantRanges(/* Bits */ 4, TestFn);
}

template <typename Fn>
static void EnumerateTwoInterestingConstantRanges(Fn TestFn) {
  for (unsigned Bits : {1, 4}) {
    EnumerateConstantRanges(Bits, [&](const ConstantRange &CR1) {
      EnumerateConstantRanges(
          Bits, [&](const ConstantRange &CR2) { TestFn(CR1, CR2); });
    });
  }
}

template <typename Fn>
static void ForeachNumInConstantRange(const ConstantRange &CR, Fn TestFn) {
  if (!CR.isEmptySet()) {
    APInt N = CR.getLower();
    do TestFn(N);
    while (++N != CR.getUpper());
  }
}

using PreferFn = llvm::function_ref<bool(const ConstantRange &,
                                         const ConstantRange &)>;

bool PreferSmallest(const ConstantRange &CR1, const ConstantRange &CR2) {
  return CR1.isSizeStrictlySmallerThan(CR2);
}

bool PreferSmallestUnsigned(const ConstantRange &CR1,
                            const ConstantRange &CR2) {
  if (CR1.isWrappedSet() != CR2.isWrappedSet())
    return CR1.isWrappedSet() < CR2.isWrappedSet();
  return PreferSmallest(CR1, CR2);
}

bool PreferSmallestSigned(const ConstantRange &CR1, const ConstantRange &CR2) {
  if (CR1.isSignWrappedSet() != CR2.isSignWrappedSet())
    return CR1.isSignWrappedSet() < CR2.isSignWrappedSet();
  return PreferSmallest(CR1, CR2);
}

bool PreferSmallestNonFullUnsigned(const ConstantRange &CR1,
                                   const ConstantRange &CR2) {
  if (CR1.isFullSet() != CR2.isFullSet())
    return CR1.isFullSet() < CR2.isFullSet();
  return PreferSmallestUnsigned(CR1, CR2);
}

bool PreferSmallestNonFullSigned(const ConstantRange &CR1,
                                 const ConstantRange &CR2) {
  if (CR1.isFullSet() != CR2.isFullSet())
    return CR1.isFullSet() < CR2.isFullSet();
  return PreferSmallestSigned(CR1, CR2);
}

testing::AssertionResult rangeContains(const ConstantRange &CR, const APInt &N,
                                       ArrayRef<ConstantRange> Inputs) {
  if (CR.contains(N))
    return testing::AssertionSuccess();

  testing::AssertionResult Result = testing::AssertionFailure();
  Result << CR << " does not contain " << N << " for inputs: ";
  for (const ConstantRange &Input : Inputs)
    Result << Input << ", ";
  return Result;
}

// Check whether constant range CR is an optimal approximation of the set
// Elems under the given PreferenceFn. The preference function should return
// true if the first range argument is strictly preferred to the second one.
static void TestRange(const ConstantRange &CR, const SmallBitVector &Elems,
                      PreferFn PreferenceFn, ArrayRef<ConstantRange> Inputs,
                      bool CheckOptimality = true) {
  unsigned BitWidth = CR.getBitWidth();

  // Check conservative correctness.
  for (unsigned Elem : Elems.set_bits()) {
    EXPECT_TRUE(rangeContains(CR, APInt(BitWidth, Elem), Inputs));
  }

  if (!CheckOptimality)
    return;

  // Make sure we have at least one element for the code below.
  if (Elems.none()) {
    EXPECT_TRUE(CR.isEmptySet());
    return;
  }

  auto NotPreferred = [&](const ConstantRange &PossibleCR) {
    if (!PreferenceFn(PossibleCR, CR))
      return testing::AssertionSuccess();

    testing::AssertionResult Result = testing::AssertionFailure();
    Result << "Inputs = ";
    for (const ConstantRange &Input : Inputs)
      Result << Input << ", ";
    Result << "CR = " << CR << ", BetterCR = " << PossibleCR;
    return Result;
  };

  // Look at all pairs of adjacent elements and the slack-free ranges
  // [Elem, PrevElem] they imply. Check that none of the ranges are strictly
  // preferred over the computed range (they may have equal preference).
  int FirstElem = Elems.find_first();
  int PrevElem = FirstElem, Elem;
  do {
    Elem = Elems.find_next(PrevElem);
    if (Elem < 0)
      Elem = FirstElem; // Wrap around to first element.

    ConstantRange PossibleCR =
        ConstantRange::getNonEmpty(APInt(BitWidth, Elem),
                                   APInt(BitWidth, PrevElem) + 1);
    // We get a full range any time PrevElem and Elem are adjacent. Avoid
    // repeated checks by skipping here, and explicitly checking below instead.
    if (!PossibleCR.isFullSet()) {
      EXPECT_TRUE(NotPreferred(PossibleCR));
    }

    PrevElem = Elem;
  } while (Elem != FirstElem);

  EXPECT_TRUE(NotPreferred(ConstantRange::getFull(BitWidth)));
}

using UnaryRangeFn = llvm::function_ref<ConstantRange(const ConstantRange &)>;
using UnaryIntFn = llvm::function_ref<std::optional<APInt>(const APInt &)>;

static void TestUnaryOpExhaustive(UnaryRangeFn RangeFn, UnaryIntFn IntFn,
                                  PreferFn PreferenceFn = PreferSmallest) {
  EnumerateInterestingConstantRanges([&](const ConstantRange &CR) {
    SmallBitVector Elems(1 << CR.getBitWidth());
    ForeachNumInConstantRange(CR, [&](const APInt &N) {
      if (std::optional<APInt> ResultN = IntFn(N))
        Elems.set(ResultN->getZExtValue());
    });
    TestRange(RangeFn(CR), Elems, PreferenceFn, {CR});
  });
}

using BinaryRangeFn = llvm::function_ref<ConstantRange(const ConstantRange &,
                                                       const ConstantRange &)>;
using BinaryIntFn =
    llvm::function_ref<std::optional<APInt>(const APInt &, const APInt &)>;
using BinaryCheckFn = llvm::function_ref<bool(const ConstantRange &,
                                              const ConstantRange &)>;

static bool CheckAll(const ConstantRange &, const ConstantRange &) {
  return true;
}

static bool CheckCorrectnessOnly(const ConstantRange &, const ConstantRange &) {
  return false;
}

static bool CheckSingleElementsOnly(const ConstantRange &CR1,
                                    const ConstantRange &CR2) {
  return CR1.isSingleElement() && CR2.isSingleElement();
}

static bool CheckNonWrappedOnly(const ConstantRange &CR1,
                                const ConstantRange &CR2) {
  return !CR1.isWrappedSet() && !CR2.isWrappedSet();
}

static bool CheckNonSignWrappedOnly(const ConstantRange &CR1,
                                    const ConstantRange &CR2) {
  return !CR1.isSignWrappedSet() && !CR2.isSignWrappedSet();
}

static bool CheckNonWrappedOrSignWrappedOnly(const ConstantRange &CR1,
                                             const ConstantRange &CR2) {
  return !CR1.isWrappedSet() && !CR1.isSignWrappedSet() &&
         !CR2.isWrappedSet() && !CR2.isSignWrappedSet();
}

// CheckFn determines whether optimality is checked for a given range pair.
// Correctness is always checked.
static void TestBinaryOpExhaustive(BinaryRangeFn RangeFn, BinaryIntFn IntFn,
                                   PreferFn PreferenceFn = PreferSmallest,
                                   BinaryCheckFn CheckFn = CheckAll) {
  EnumerateTwoInterestingConstantRanges(
      [&](const ConstantRange &CR1, const ConstantRange &CR2) {
        SmallBitVector Elems(1 << CR1.getBitWidth());
        ForeachNumInConstantRange(CR1, [&](const APInt &N1) {
          ForeachNumInConstantRange(CR2, [&](const APInt &N2) {
            if (std::optional<APInt> ResultN = IntFn(N1, N2))
              Elems.set(ResultN->getZExtValue());
          });
        });
        TestRange(RangeFn(CR1, CR2), Elems, PreferenceFn, {CR1, CR2},
                  CheckFn(CR1, CR2));
      });
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

template <typename Fn1, typename Fn2, typename Fn3>
void testBinarySetOperationExhaustive(Fn1 OpFn, Fn2 ExactOpFn, Fn3 InResultFn) {
  EnumerateTwoInterestingConstantRanges(
      [=](const ConstantRange &CR1, const ConstantRange &CR2) {
        unsigned Bits = CR1.getBitWidth();
        SmallBitVector Elems(1 << Bits);
        APInt Num(Bits, 0);
        for (unsigned I = 0, Limit = 1 << Bits; I < Limit; ++I, ++Num)
          if (InResultFn(CR1, CR2, Num))
            Elems.set(Num.getZExtValue());

        ConstantRange SmallestCR = OpFn(CR1, CR2, ConstantRange::Smallest);
        TestRange(SmallestCR, Elems, PreferSmallest, {CR1, CR2});

        ConstantRange UnsignedCR = OpFn(CR1, CR2, ConstantRange::Unsigned);
        TestRange(UnsignedCR, Elems, PreferSmallestNonFullUnsigned, {CR1, CR2});

        ConstantRange SignedCR = OpFn(CR1, CR2, ConstantRange::Signed);
        TestRange(SignedCR, Elems, PreferSmallestNonFullSigned, {CR1, CR2});

        std::optional<ConstantRange> ExactCR = ExactOpFn(CR1, CR2);
        if (SmallestCR.isSizeLargerThan(Elems.count())) {
          EXPECT_TRUE(!ExactCR);
        } else {
          EXPECT_EQ(SmallestCR, *ExactCR);
        }
      });
}

TEST_F(ConstantRangeTest, IntersectWithExhaustive) {
  testBinarySetOperationExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2,
         ConstantRange::PreferredRangeType Type) {
        return CR1.intersectWith(CR2, Type);
      },
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.exactIntersectWith(CR2);
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
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.exactUnionWith(CR2);
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

TEST_F(ConstantRangeTest, getActiveBits) {
  EnumerateInterestingConstantRanges([&](const ConstantRange &CR) {
    unsigned Exact = 0;
    ForeachNumInConstantRange(CR, [&](const APInt &N) {
      Exact = std::max(Exact, N.getActiveBits());
    });

    unsigned ResultCR = CR.getActiveBits();
    EXPECT_EQ(Exact, ResultCR);
  });
}
TEST_F(ConstantRangeTest, losslessUnsignedTruncationZeroext) {
  EnumerateInterestingConstantRanges([&](const ConstantRange &CR) {
    unsigned Bits = CR.getBitWidth();
    unsigned MinBitWidth = CR.getActiveBits();
    if (MinBitWidth == 0) {
      EXPECT_TRUE(CR.isEmptySet() ||
                  (CR.isSingleElement() && CR.getSingleElement()->isZero()));
      return;
    }
    if (MinBitWidth == Bits)
      return;
    EXPECT_EQ(CR, CR.truncate(MinBitWidth).zeroExtend(Bits));
  });
}

TEST_F(ConstantRangeTest, getMinSignedBits) {
  EnumerateInterestingConstantRanges([&](const ConstantRange &CR) {
    unsigned Exact = 0;
    ForeachNumInConstantRange(CR, [&](const APInt &N) {
      Exact = std::max(Exact, N.getSignificantBits());
    });

    unsigned ResultCR = CR.getMinSignedBits();
    EXPECT_EQ(Exact, ResultCR);
  });
}
TEST_F(ConstantRangeTest, losslessSignedTruncationSignext) {
  EnumerateInterestingConstantRanges([&](const ConstantRange &CR) {
    unsigned Bits = CR.getBitWidth();
    unsigned MinBitWidth = CR.getMinSignedBits();
    if (MinBitWidth == 0) {
      EXPECT_TRUE(CR.isEmptySet());
      return;
    }
    if (MinBitWidth == Bits)
      return;
    EXPECT_EQ(CR, CR.truncate(MinBitWidth).signExtend(Bits));
  });
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

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.add(CR2);
      },
      [](const APInt &N1, const APInt &N2) {
        return N1 + N2;
      });
}

TEST_F(ConstantRangeTest, AddWithNoWrap) {
  typedef OverflowingBinaryOperator OBO;
  EXPECT_EQ(Empty.addWithNoWrap(Some, OBO::NoSignedWrap), Empty);
  EXPECT_EQ(Some.addWithNoWrap(Empty, OBO::NoSignedWrap), Empty);
  EXPECT_EQ(Full.addWithNoWrap(Full, OBO::NoSignedWrap), Full);
  EXPECT_NE(Full.addWithNoWrap(Some, OBO::NoSignedWrap), Full);
  EXPECT_NE(Some.addWithNoWrap(Full, OBO::NoSignedWrap), Full);
  EXPECT_EQ(Full.addWithNoWrap(ConstantRange(APInt(16, 1), APInt(16, 2)),
                               OBO::NoSignedWrap),
            ConstantRange(APInt(16, INT16_MIN + 1), APInt(16, INT16_MIN)));
  EXPECT_EQ(ConstantRange(APInt(16, 1), APInt(16, 2))
                .addWithNoWrap(Full, OBO::NoSignedWrap),
            ConstantRange(APInt(16, INT16_MIN + 1), APInt(16, INT16_MIN)));
  EXPECT_EQ(Full.addWithNoWrap(ConstantRange(APInt(16, -1), APInt(16, 0)),
                               OBO::NoSignedWrap),
            ConstantRange(APInt(16, INT16_MIN), APInt(16, INT16_MAX)));
  EXPECT_EQ(ConstantRange(APInt(8, 100), APInt(8, 120))
                .addWithNoWrap(ConstantRange(APInt(8, 120), APInt(8, 123)),
                               OBO::NoSignedWrap),
            ConstantRange(8, false));
  EXPECT_EQ(ConstantRange(APInt(8, -120), APInt(8, -100))
                .addWithNoWrap(ConstantRange(APInt(8, -110), APInt(8, -100)),
                               OBO::NoSignedWrap),
            ConstantRange(8, false));
  EXPECT_EQ(ConstantRange(APInt(8, 0), APInt(8, 101))
                .addWithNoWrap(ConstantRange(APInt(8, -128), APInt(8, 28)),
                               OBO::NoSignedWrap),
            ConstantRange(8, true));
  EXPECT_EQ(ConstantRange(APInt(8, 0), APInt(8, 101))
                .addWithNoWrap(ConstantRange(APInt(8, -120), APInt(8, 29)),
                               OBO::NoSignedWrap),
            ConstantRange(APInt(8, -120), APInt(8, -128)));
  EXPECT_EQ(ConstantRange(APInt(8, -50), APInt(8, 50))
                .addWithNoWrap(ConstantRange(APInt(8, 10), APInt(8, 20)),
                               OBO::NoSignedWrap),
            ConstantRange(APInt(8, -40), APInt(8, 69)));
  EXPECT_EQ(ConstantRange(APInt(8, 10), APInt(8, 20))
                .addWithNoWrap(ConstantRange(APInt(8, -50), APInt(8, 50)),
                               OBO::NoSignedWrap),
            ConstantRange(APInt(8, -40), APInt(8, 69)));
  EXPECT_EQ(ConstantRange(APInt(8, 120), APInt(8, -10))
                .addWithNoWrap(ConstantRange(APInt(8, 5), APInt(8, 20)),
                               OBO::NoSignedWrap),
            ConstantRange(APInt(8, 125), APInt(8, 9)));
  EXPECT_EQ(ConstantRange(APInt(8, 5), APInt(8, 20))
                .addWithNoWrap(ConstantRange(APInt(8, 120), APInt(8, -10)),
                               OBO::NoSignedWrap),
            ConstantRange(APInt(8, 125), APInt(8, 9)));

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.addWithNoWrap(CR2, OBO::NoSignedWrap);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        bool IsOverflow;
        APInt Res = N1.sadd_ov(N2, IsOverflow);
        if (IsOverflow)
          return std::nullopt;
        return Res;
      },
      PreferSmallest, CheckNonSignWrappedOnly);

  EXPECT_EQ(Empty.addWithNoWrap(Some, OBO::NoUnsignedWrap), Empty);
  EXPECT_EQ(Some.addWithNoWrap(Empty, OBO::NoUnsignedWrap), Empty);
  EXPECT_EQ(Full.addWithNoWrap(Full, OBO::NoUnsignedWrap), Full);
  EXPECT_NE(Full.addWithNoWrap(Some, OBO::NoUnsignedWrap), Full);
  EXPECT_NE(Some.addWithNoWrap(Full, OBO::NoUnsignedWrap), Full);
  EXPECT_EQ(Full.addWithNoWrap(ConstantRange(APInt(16, 1), APInt(16, 2)),
                               OBO::NoUnsignedWrap),
            ConstantRange(APInt(16, 1), APInt(16, 0)));
  EXPECT_EQ(ConstantRange(APInt(16, 1), APInt(16, 2))
                .addWithNoWrap(Full, OBO::NoUnsignedWrap),
            ConstantRange(APInt(16, 1), APInt(16, 0)));
  EXPECT_EQ(ConstantRange(APInt(8, 200), APInt(8, 220))
                .addWithNoWrap(ConstantRange(APInt(8, 100), APInt(8, 123)),
                               OBO::NoUnsignedWrap),
            ConstantRange(8, false));
  EXPECT_EQ(ConstantRange(APInt(8, 0), APInt(8, 101))
                .addWithNoWrap(ConstantRange(APInt(8, 0), APInt(8, 156)),
                               OBO::NoUnsignedWrap),
            ConstantRange(8, true));
  EXPECT_EQ(ConstantRange(APInt(8, 0), APInt(8, 101))
                .addWithNoWrap(ConstantRange(APInt(8, 10), APInt(8, 29)),
                               OBO::NoUnsignedWrap),
            ConstantRange(APInt(8, 10), APInt(8, 129)));
  EXPECT_EQ(ConstantRange(APInt(8, 20), APInt(8, 10))
                .addWithNoWrap(ConstantRange(APInt(8, 50), APInt(8, 200)),
                               OBO::NoUnsignedWrap),
            ConstantRange(APInt(8, 50), APInt(8, 0)));
  EXPECT_EQ(ConstantRange(APInt(8, 10), APInt(8, 20))
                .addWithNoWrap(ConstantRange(APInt(8, 50), APInt(8, 200)),
                               OBO::NoUnsignedWrap),
            ConstantRange(APInt(8, 60), APInt(8, -37)));
  EXPECT_EQ(ConstantRange(APInt(8, 20), APInt(8, -30))
                .addWithNoWrap(ConstantRange(APInt(8, 5), APInt(8, 20)),
                               OBO::NoUnsignedWrap),
            ConstantRange(APInt(8, 25), APInt(8, -11)));
  EXPECT_EQ(ConstantRange(APInt(8, 5), APInt(8, 20))
                .addWithNoWrap(ConstantRange(APInt(8, 20), APInt(8, -30)),
                               OBO::NoUnsignedWrap),
            ConstantRange(APInt(8, 25), APInt(8, -11)));

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.addWithNoWrap(CR2, OBO::NoUnsignedWrap);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        bool IsOverflow;
        APInt Res = N1.uadd_ov(N2, IsOverflow);
        if (IsOverflow)
          return std::nullopt;
        return Res;
      },
      PreferSmallest, CheckNonWrappedOnly);

  EXPECT_EQ(ConstantRange(APInt(8, 50), APInt(8, 100))
                .addWithNoWrap(ConstantRange(APInt(8, 20), APInt(8, 70)),
                               OBO::NoSignedWrap),
            ConstantRange(APInt(8, 70), APInt(8, -128)));
  EXPECT_EQ(ConstantRange(APInt(8, 50), APInt(8, 100))
                .addWithNoWrap(ConstantRange(APInt(8, 20), APInt(8, 70)),
                               OBO::NoUnsignedWrap),
            ConstantRange(APInt(8, 70), APInt(8, 169)));
  EXPECT_EQ(ConstantRange(APInt(8, 50), APInt(8, 100))
                .addWithNoWrap(ConstantRange(APInt(8, 20), APInt(8, 70)),
                               OBO::NoUnsignedWrap | OBO::NoSignedWrap),
            ConstantRange(APInt(8, 70), APInt(8, -128)));

  EXPECT_EQ(ConstantRange(APInt(8, -100), APInt(8, -50))
                .addWithNoWrap(ConstantRange(APInt(8, 20), APInt(8, 30)),
                               OBO::NoSignedWrap),
            ConstantRange(APInt(8, -80), APInt(8, -21)));
  EXPECT_EQ(ConstantRange(APInt(8, -100), APInt(8, -50))
                .addWithNoWrap(ConstantRange(APInt(8, 20), APInt(8, 30)),
                               OBO::NoUnsignedWrap),
            ConstantRange(APInt(8, 176), APInt(8, 235)));
  EXPECT_EQ(ConstantRange(APInt(8, -100), APInt(8, -50))
                .addWithNoWrap(ConstantRange(APInt(8, 20), APInt(8, 30)),
                               OBO::NoUnsignedWrap | OBO::NoSignedWrap),
            ConstantRange(APInt(8, 176), APInt(8, 235)));

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.addWithNoWrap(CR2, OBO::NoUnsignedWrap | OBO::NoSignedWrap);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        bool IsOverflow1, IsOverflow2;
        APInt Res1 = N1.uadd_ov(N2, IsOverflow1);
        APInt Res2 = N1.sadd_ov(N2, IsOverflow2);
        if (IsOverflow1 || IsOverflow2)
          return std::nullopt;
        assert(Res1 == Res2 && "Addition results differ?");
        return Res1;
      },
      PreferSmallest, CheckNonWrappedOrSignWrappedOnly);
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

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.sub(CR2);
      },
      [](const APInt &N1, const APInt &N2) {
        return N1 - N2;
      });
}

TEST_F(ConstantRangeTest, SubWithNoWrap) {
  typedef OverflowingBinaryOperator OBO;
  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.subWithNoWrap(CR2, OBO::NoSignedWrap);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        bool IsOverflow;
        APInt Res = N1.ssub_ov(N2, IsOverflow);
        if (IsOverflow)
          return std::nullopt;
        return Res;
      },
      PreferSmallest, CheckNonSignWrappedOnly);
  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.subWithNoWrap(CR2, OBO::NoUnsignedWrap);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        bool IsOverflow;
        APInt Res = N1.usub_ov(N2, IsOverflow);
        if (IsOverflow)
          return std::nullopt;
        return Res;
      },
      PreferSmallest, CheckNonWrappedOnly);
  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.subWithNoWrap(CR2, OBO::NoUnsignedWrap | OBO::NoSignedWrap);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        bool IsOverflow1, IsOverflow2;
        APInt Res1 = N1.usub_ov(N2, IsOverflow1);
        APInt Res2 = N1.ssub_ov(N2, IsOverflow2);
        if (IsOverflow1 || IsOverflow2)
          return std::nullopt;
        assert(Res1 == Res2 && "Subtraction results differ?");
        return Res1;
      },
      PreferSmallest, CheckNonWrappedOrSignWrappedOnly);
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

  // Multiplication by -1 should give precise results.
  EXPECT_EQ(ConstantRange(APInt(8, 3), APInt(8, -11))
                .multiply(ConstantRange(APInt(8, -1))),
            ConstantRange(APInt(8, 12), APInt(8, -2)));
  EXPECT_EQ(ConstantRange(APInt(8, -1))
                .multiply(ConstantRange(APInt(8, 3), APInt(8, -11))),
            ConstantRange(APInt(8, 12), APInt(8, -2)));

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.multiply(CR2);
      },
      [](const APInt &N1, const APInt &N2) {
        return N1 * N2;
      },
      PreferSmallest,
      [](const ConstantRange &, const ConstantRange &) {
        return false; // Check correctness only.
      });
}

TEST_F(ConstantRangeTest, MultiplyWithNoWrap) {
  using OBO = OverflowingBinaryOperator;

  EXPECT_EQ(Empty.multiplyWithNoWrap(Some, OBO::NoUnsignedWrap), Empty);
  EXPECT_EQ(Some.multiplyWithNoWrap(Empty, OBO::NoUnsignedWrap), Empty);
  EXPECT_EQ(Full.multiplyWithNoWrap(Full, OBO::NoUnsignedWrap), Full);
  EXPECT_EQ(Full.multiplyWithNoWrap(Some, OBO::NoUnsignedWrap), Full);
  EXPECT_EQ(Some.multiplyWithNoWrap(Full, OBO::NoUnsignedWrap), Full);
  EXPECT_EQ(ConstantRange(APInt(4, 0), APInt(4, 2))
                .multiplyWithNoWrap(ConstantRange(APInt(4, 2), APInt(4, 0)),
                                    OBO::NoUnsignedWrap),
            ConstantRange::getFull(4));
  EXPECT_EQ(ConstantRange(APInt(4, 1), APInt(4, 5))
                .multiplyWithNoWrap(ConstantRange(APInt(4, 1), APInt(4, 5)),
                                    OBO::NoUnsignedWrap),
            ConstantRange(APInt(4, 1), APInt(4, 0)));
  EXPECT_EQ(ConstantRange(APInt(8, 254), APInt(8, 0))
                .multiplyWithNoWrap(ConstantRange(APInt(8, 252), APInt(8, 4)),
                                    OBO::NoUnsignedWrap),
            ConstantRange(APInt(8, 250), APInt(8, 9)));
  EXPECT_EQ(ConstantRange(APInt(8, 254), APInt(8, 255))
                .multiplyWithNoWrap(ConstantRange(APInt(8, 2), APInt(8, 4)),
                                    OBO::NoUnsignedWrap),
            ConstantRange::getEmpty(8));

  EXPECT_EQ(Empty.multiplyWithNoWrap(Some, OBO::NoSignedWrap), Empty);
  EXPECT_EQ(Some.multiplyWithNoWrap(Empty, OBO::NoSignedWrap), Empty);
  EXPECT_EQ(Full.multiplyWithNoWrap(Full, OBO::NoSignedWrap), Full);
  EXPECT_EQ(Full.multiplyWithNoWrap(Some, OBO::NoSignedWrap), Full);
  EXPECT_EQ(Some.multiplyWithNoWrap(Full, OBO::NoSignedWrap), Full);
  EXPECT_EQ(
      ConstantRange(APInt(4, 0), APInt(4, 4))
          .multiplyWithNoWrap(ConstantRange(APInt(4, -5, true), APInt(4, 4)),
                              OBO::NoSignedWrap),
      ConstantRange::getFull(4));
  EXPECT_EQ(ConstantRange(APInt(4, 0), APInt(4, 3))
                .multiplyWithNoWrap(ConstantRange(APInt(4, 0), APInt(4, 5)),
                                    OBO::NoSignedWrap),
            ConstantRange(APInt(4, 0), APInt(4, -8, true)));
  EXPECT_EQ(ConstantRange(APInt(8, 3), APInt(8, -11, true))
                .multiplyWithNoWrap(ConstantRange(APInt(8, -1, true)),
                                    OBO::NoSignedWrap),
            ConstantRange(APInt(8, 12), APInt(8, -2, true)));
  EXPECT_EQ(ConstantRange(APInt(8, 254), APInt(8, 255))
                .multiplyWithNoWrap(ConstantRange(APInt(8, 100), APInt(8, 121)),
                                    OBO::NoSignedWrap),
            ConstantRange::getEmpty(8));

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.multiplyWithNoWrap(CR2, OBO::NoUnsignedWrap);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        bool IsOverflow;
        APInt Res = N1.umul_ov(N2, IsOverflow);
        if (IsOverflow)
          return std::nullopt;
        return Res;
      },
      PreferSmallest, CheckCorrectnessOnly);
  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.multiplyWithNoWrap(CR2, OBO::NoSignedWrap);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        bool IsOverflow;
        APInt Res = N1.smul_ov(N2, IsOverflow);
        if (IsOverflow)
          return std::nullopt;
        return Res;
      },
      PreferSmallest, CheckCorrectnessOnly);
  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.multiplyWithNoWrap(CR2,
                                      OBO::NoUnsignedWrap | OBO::NoSignedWrap);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        bool IsOverflow1, IsOverflow2;
        APInt Res1 = N1.umul_ov(N2, IsOverflow1);
        APInt Res2 = N1.smul_ov(N2, IsOverflow2);
        if (IsOverflow1 || IsOverflow2)
          return std::nullopt;
        assert(Res1 == Res2 && "Multiplication results differ?");
        return Res1;
      },
      PreferSmallest, CheckCorrectnessOnly);
}

TEST_F(ConstantRangeTest, smul_fast) {
  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.smul_fast(CR2);
      },
      [](const APInt &N1, const APInt &N2) { return N1 * N2; }, PreferSmallest,
      CheckCorrectnessOnly);
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
  EXPECT_EQ(Wrap.umax(Wrap), Wrap);
  EXPECT_EQ(Wrap.umax(One), ConstantRange(APInt(16, 0xa), APInt(16, 0)));
  EXPECT_EQ(One.umax(One), One);

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.umax(CR2);
      },
      [](const APInt &N1, const APInt &N2) {
        return APIntOps::umax(N1, N2);
      },
      PreferSmallestNonFullUnsigned);
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

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.smax(CR2);
      },
      [](const APInt &N1, const APInt &N2) {
        return APIntOps::smax(N1, N2);
      },
      PreferSmallestNonFullSigned);
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
  EXPECT_EQ(Wrap.umin(Wrap), Wrap);
  EXPECT_EQ(Wrap.umin(One), ConstantRange(APInt(16, 0), APInt(16, 0xb)));
  EXPECT_EQ(One.umin(One), One);

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.umin(CR2);
      },
      [](const APInt &N1, const APInt &N2) {
        return APIntOps::umin(N1, N2);
      },
      PreferSmallestNonFullUnsigned);
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
  EXPECT_EQ(Wrap.smin(Wrap), Wrap);
  EXPECT_EQ(Wrap.smin(One), ConstantRange(APInt(16, (uint64_t)INT16_MIN),
                                          APInt(16, 0xb)));
  EXPECT_EQ(One.smin(One), One);

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.smin(CR2);
      },
      [](const APInt &N1, const APInt &N2) {
        return APIntOps::smin(N1, N2);
      },
      PreferSmallestNonFullSigned);
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


  ConstantRange Zero(APInt(16, 0));
  EXPECT_EQ(Zero.udiv(One), Zero);
  EXPECT_EQ(Zero.udiv(Full), Zero);

  EXPECT_EQ(ConstantRange(APInt(16, 0), APInt(16, 99)).udiv(Full),
            ConstantRange(APInt(16, 0), APInt(16, 99)));
  EXPECT_EQ(ConstantRange(APInt(16, 10), APInt(16, 99)).udiv(Full),
            ConstantRange(APInt(16, 0), APInt(16, 99)));
}

TEST_F(ConstantRangeTest, SDiv) {
  ConstantRange OneBit = ConstantRange::getFull(1);
  EXPECT_EQ(OneBit.sdiv(OneBit), ConstantRange(APInt(1, 0)));

  EnumerateTwoInterestingConstantRanges([&](const ConstantRange &CR1,
                                            const ConstantRange &CR2) {
    // Collect possible results in a bit vector. We store the signed value plus
    // a bias to make it unsigned.
    unsigned Bits = CR1.getBitWidth();
    int Bias = 1 << (Bits - 1);
    BitVector Results(1 << Bits);
    ForeachNumInConstantRange(CR1, [&](const APInt &N1) {
      ForeachNumInConstantRange(CR2, [&](const APInt &N2) {
        // Division by zero is UB.
        if (N2 == 0)
          return;

        // SignedMin / -1 is UB.
        if (N1.isMinSignedValue() && N2.isAllOnes())
          return;

        APInt N = N1.sdiv(N2);
        Results.set(N.getSExtValue() + Bias);
      });
    });

    ConstantRange CR = CR1.sdiv(CR2);
    if (Results.none()) {
      EXPECT_TRUE(CR.isEmptySet());
      return;
    }

    // If there is a non-full signed envelope, that should be the result.
    APInt SMin(Bits, Results.find_first() - Bias);
    APInt SMax(Bits, Results.find_last() - Bias);
    ConstantRange Envelope = ConstantRange::getNonEmpty(SMin, SMax + 1);
    if (!Envelope.isFullSet()) {
      EXPECT_EQ(Envelope, CR);
      return;
    }

    // If the signed envelope is a full set, try to find a smaller sign wrapped
    // set that is separated in negative and positive components (or one which
    // can also additionally contain zero).
    int LastNeg = Results.find_last_in(0, Bias) - Bias;
    int LastPos = Results.find_next(Bias) - Bias;
    if (Results[Bias]) {
      if (LastNeg == -1)
        ++LastNeg;
      else if (LastPos == 1)
        --LastPos;
    }

    APInt WMax(Bits, LastNeg);
    APInt WMin(Bits, LastPos);
    ConstantRange Wrapped = ConstantRange::getNonEmpty(WMin, WMax + 1);
    EXPECT_EQ(Wrapped, CR);
  });
}

TEST_F(ConstantRangeTest, URem) {
  EXPECT_EQ(Full.urem(Empty), Empty);
  EXPECT_EQ(Empty.urem(Full), Empty);
  // urem by zero is poison.
  EXPECT_EQ(Full.urem(ConstantRange(APInt(16, 0))), Empty);
  // urem by full range doesn't contain MaxValue.
  EXPECT_EQ(Full.urem(Full), ConstantRange(APInt(16, 0), APInt(16, 0xffff)));
  // urem is upper bounded by maximum RHS minus one.
  EXPECT_EQ(Full.urem(ConstantRange(APInt(16, 0), APInt(16, 123))),
            ConstantRange(APInt(16, 0), APInt(16, 122)));
  // urem is upper bounded by maximum LHS.
  EXPECT_EQ(ConstantRange(APInt(16, 0), APInt(16, 123)).urem(Full),
            ConstantRange(APInt(16, 0), APInt(16, 123)));
  // If the LHS is always lower than the RHS, the result is the LHS.
  EXPECT_EQ(ConstantRange(APInt(16, 10), APInt(16, 20))
                .urem(ConstantRange(APInt(16, 20), APInt(16, 30))),
            ConstantRange(APInt(16, 10), APInt(16, 20)));
  // It has to be strictly lower, otherwise the top value may wrap to zero.
  EXPECT_EQ(ConstantRange(APInt(16, 10), APInt(16, 20))
                .urem(ConstantRange(APInt(16, 19), APInt(16, 30))),
            ConstantRange(APInt(16, 0), APInt(16, 20)));
  // [12, 14] % 10 is [2, 4], but we conservatively compute [0, 9].
  EXPECT_EQ(ConstantRange(APInt(16, 12), APInt(16, 15))
                .urem(ConstantRange(APInt(16, 10))),
            ConstantRange(APInt(16, 0), APInt(16, 10)));

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.urem(CR2);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        if (N2.isZero())
          return std::nullopt;
        return N1.urem(N2);
      },
      PreferSmallest, CheckSingleElementsOnly);
}

TEST_F(ConstantRangeTest, SRem) {
  EXPECT_EQ(Full.srem(Empty), Empty);
  EXPECT_EQ(Empty.srem(Full), Empty);
  // srem by zero is UB.
  EXPECT_EQ(Full.srem(ConstantRange(APInt(16, 0))), Empty);
  // srem by full range doesn't contain SignedMinValue.
  EXPECT_EQ(Full.srem(Full), ConstantRange(APInt::getSignedMinValue(16) + 1,
                                           APInt::getSignedMinValue(16)));

  ConstantRange PosMod(APInt(16, 10), APInt(16, 21));  // [10, 20]
  ConstantRange NegMod(APInt(16, -20), APInt(16, -9)); // [-20, -10]
  ConstantRange IntMinMod(APInt::getSignedMinValue(16));

  ConstantRange Expected(16, true);

  // srem is bounded by abs(RHS) minus one.
  ConstantRange PosLargeLHS(APInt(16, 0), APInt(16, 41));
  Expected = ConstantRange(APInt(16, 0), APInt(16, 20));
  EXPECT_EQ(PosLargeLHS.srem(PosMod), Expected);
  EXPECT_EQ(PosLargeLHS.srem(NegMod), Expected);
  ConstantRange NegLargeLHS(APInt(16, -40), APInt(16, 1));
  Expected = ConstantRange(APInt(16, -19), APInt(16, 1));
  EXPECT_EQ(NegLargeLHS.srem(PosMod), Expected);
  EXPECT_EQ(NegLargeLHS.srem(NegMod), Expected);
  ConstantRange PosNegLargeLHS(APInt(16, -32), APInt(16, 38));
  Expected = ConstantRange(APInt(16, -19), APInt(16, 20));
  EXPECT_EQ(PosNegLargeLHS.srem(PosMod), Expected);
  EXPECT_EQ(PosNegLargeLHS.srem(NegMod), Expected);

  // srem is bounded by LHS.
  ConstantRange PosLHS(APInt(16, 0), APInt(16, 16));
  EXPECT_EQ(PosLHS.srem(PosMod), PosLHS);
  EXPECT_EQ(PosLHS.srem(NegMod), PosLHS);
  EXPECT_EQ(PosLHS.srem(IntMinMod), PosLHS);
  ConstantRange NegLHS(APInt(16, -15), APInt(16, 1));
  EXPECT_EQ(NegLHS.srem(PosMod), NegLHS);
  EXPECT_EQ(NegLHS.srem(NegMod), NegLHS);
  EXPECT_EQ(NegLHS.srem(IntMinMod), NegLHS);
  ConstantRange PosNegLHS(APInt(16, -12), APInt(16, 18));
  EXPECT_EQ(PosNegLHS.srem(PosMod), PosNegLHS);
  EXPECT_EQ(PosNegLHS.srem(NegMod), PosNegLHS);
  EXPECT_EQ(PosNegLHS.srem(IntMinMod), PosNegLHS);

  // srem is LHS if it is smaller than RHS.
  ConstantRange PosSmallLHS(APInt(16, 3), APInt(16, 8));
  EXPECT_EQ(PosSmallLHS.srem(PosMod), PosSmallLHS);
  EXPECT_EQ(PosSmallLHS.srem(NegMod), PosSmallLHS);
  EXPECT_EQ(PosSmallLHS.srem(IntMinMod), PosSmallLHS);
  ConstantRange NegSmallLHS(APInt(16, -7), APInt(16, -2));
  EXPECT_EQ(NegSmallLHS.srem(PosMod), NegSmallLHS);
  EXPECT_EQ(NegSmallLHS.srem(NegMod), NegSmallLHS);
  EXPECT_EQ(NegSmallLHS.srem(IntMinMod), NegSmallLHS);
  ConstantRange PosNegSmallLHS(APInt(16, -3), APInt(16, 8));
  EXPECT_EQ(PosNegSmallLHS.srem(PosMod), PosNegSmallLHS);
  EXPECT_EQ(PosNegSmallLHS.srem(NegMod), PosNegSmallLHS);
  EXPECT_EQ(PosNegSmallLHS.srem(IntMinMod), PosNegSmallLHS);

  // Example of a suboptimal result:
  // [12, 14] srem 10 is [2, 4], but we conservatively compute [0, 9].
  EXPECT_EQ(ConstantRange(APInt(16, 12), APInt(16, 15))
                .srem(ConstantRange(APInt(16, 10))),
            ConstantRange(APInt(16, 0), APInt(16, 10)));

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.srem(CR2);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        if (N2.isZero())
          return std::nullopt;
        return N1.srem(N2);
      },
      PreferSmallest, CheckSingleElementsOnly);
}

TEST_F(ConstantRangeTest, Shl) {
  ConstantRange Some2(APInt(16, 0xfff), APInt(16, 0x8000));
  ConstantRange WrapNullMax(APInt(16, 0x1), APInt(16, 0x0));
  EXPECT_EQ(Full.shl(Full), Full);
  EXPECT_EQ(Full.shl(Empty), Empty);
  EXPECT_EQ(Full.shl(One), ConstantRange(APInt(16, 0),
                                         APInt(16, 0xfc00) + 1));
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

  ConstantRange NegOne(APInt(16, 0xffff));
  EXPECT_EQ(NegOne.shl(ConstantRange(APInt(16, 0), APInt(16, 5))),
            ConstantRange(APInt(16, 0xfff0), APInt(16, 0)));
  EXPECT_EQ(ConstantRange(APInt(16, 0xfffe), APInt(16, 0))
                .shl(ConstantRange(APInt(16, 0), APInt(16, 5))),
            ConstantRange(APInt(16, 0xffe0), APInt(16, 0)));

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.shl(CR2);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        if (N2.uge(N2.getBitWidth()))
          return std::nullopt;
        return N1.shl(N2);
      },
      PreferSmallestUnsigned,
      [](const ConstantRange &, const ConstantRange &CR2) {
        // We currently only produce precise results for single element RHS.
        return CR2.isSingleElement();
      });
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

void ICmpTestImpl(CmpInst::Predicate Pred) {
  EnumerateTwoInterestingConstantRanges(
      [&](const ConstantRange &CR1, const ConstantRange &CR2) {
        bool Exhaustive = true;
        ForeachNumInConstantRange(CR1, [&](const APInt &N1) {
          ForeachNumInConstantRange(CR2, [&](const APInt &N2) {
            Exhaustive &= ICmpInst::compare(N1, N2, Pred);
          });
        });
        EXPECT_EQ(CR1.icmp(Pred, CR2), Exhaustive);
      });
}

TEST(ConstantRange, ICmp) {
  for (auto Pred : ICmpInst::predicates())
    ICmpTestImpl(Pred);
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

  ConstantRange OneLessThanBitWidth(APInt(32, 0), APInt(32, 31) + 1);
  ConstantRange UpToBitWidth(APInt(32, 0), APInt(32, 32) + 1);
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Shl, UpToBitWidth, OBO::NoUnsignedWrap),
            ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Shl, OneLessThanBitWidth, OBO::NoUnsignedWrap));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Shl, UpToBitWidth, OBO::NoSignedWrap),
            ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Shl, OneLessThanBitWidth, OBO::NoSignedWrap));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Shl, UpToBitWidth, OBO::NoUnsignedWrap),
            ConstantRange(APInt(32, 0), APInt(32, 1) + 1));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Shl, UpToBitWidth, OBO::NoSignedWrap),
            ConstantRange(APInt(32, -1), APInt(32, 0) + 1));

  EXPECT_EQ(
      ConstantRange::makeGuaranteedNoWrapRegion(
          Instruction::Shl, ConstantRange::getFull(32), OBO::NoUnsignedWrap),
      ConstantRange::makeGuaranteedNoWrapRegion(
          Instruction::Shl, OneLessThanBitWidth, OBO::NoUnsignedWrap));
  EXPECT_EQ(
      ConstantRange::makeGuaranteedNoWrapRegion(
          Instruction::Shl, ConstantRange::getFull(32), OBO::NoSignedWrap),
      ConstantRange::makeGuaranteedNoWrapRegion(
          Instruction::Shl, OneLessThanBitWidth, OBO::NoSignedWrap));

  ConstantRange IllegalShAmt(APInt(32, 32), APInt(32, 0) + 1);
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Shl, IllegalShAmt, OBO::NoUnsignedWrap),
            ConstantRange::getFull(32));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Shl, IllegalShAmt, OBO::NoSignedWrap),
            ConstantRange::getFull(32));

  EXPECT_EQ(
      ConstantRange::makeGuaranteedNoWrapRegion(
          Instruction::Shl, ConstantRange(APInt(32, -32), APInt(32, 16) + 1),
          OBO::NoUnsignedWrap),
      ConstantRange::makeGuaranteedNoWrapRegion(
          Instruction::Shl, ConstantRange(APInt(32, 0), APInt(32, 16) + 1),
          OBO::NoUnsignedWrap));
  EXPECT_EQ(
      ConstantRange::makeGuaranteedNoWrapRegion(
          Instruction::Shl, ConstantRange(APInt(32, -32), APInt(32, 16) + 1),
          OBO::NoSignedWrap),
      ConstantRange::makeGuaranteedNoWrapRegion(
          Instruction::Shl, ConstantRange(APInt(32, 0), APInt(32, 16) + 1),
          OBO::NoSignedWrap));

  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Shl,
                ConstantRange(APInt(32, -32), APInt(32, 16) + 1),
                OBO::NoUnsignedWrap),
            ConstantRange(APInt(32, 0), APInt(32, 65535) + 1));
  EXPECT_EQ(ConstantRange::makeGuaranteedNoWrapRegion(
                Instruction::Shl,
                ConstantRange(APInt(32, -32), APInt(32, 16) + 1),
                OBO::NoSignedWrap),
            ConstantRange(APInt(32, -32768), APInt(32, 32767) + 1));
}

template<typename Fn>
void TestNoWrapRegionExhaustive(Instruction::BinaryOps BinOp,
                                unsigned NoWrapKind, Fn OverflowFn) {
  for (unsigned Bits : {1, 5}) {
    EnumerateConstantRanges(Bits, [&](const ConstantRange &CR) {
      if (CR.isEmptySet())
        return;
      if (Instruction::isShift(BinOp) && CR.getUnsignedMax().uge(Bits))
        return;

      ConstantRange NoWrap =
          ConstantRange::makeGuaranteedNoWrapRegion(BinOp, CR, NoWrapKind);
      EnumerateAPInts(Bits, [&](const APInt &N1) {
        bool NoOverflow = true;
        bool Overflow = true;
        ForeachNumInConstantRange(CR, [&](const APInt &N2) {
          if (OverflowFn(N1, N2))
            NoOverflow = false;
          else
            Overflow = false;
        });
        EXPECT_EQ(NoOverflow, NoWrap.contains(N1));

        // The no-wrap range is exact for single-element ranges.
        if (CR.isSingleElement()) {
          EXPECT_EQ(Overflow, !NoWrap.contains(N1));
        }
      });
    });
  }
}

// Show that makeGuaranteedNoWrapRegion() is maximal, and for single-element
// ranges also exact.
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
  TestNoWrapRegionExhaustive(Instruction::Shl,
                             OverflowingBinaryOperator::NoUnsignedWrap,
                             [](const APInt &N1, const APInt &N2) {
                               bool Overflow;
                               (void)N1.ushl_ov(N2, Overflow);
                               return Overflow;
                             });
  TestNoWrapRegionExhaustive(Instruction::Shl,
                             OverflowingBinaryOperator::NoSignedWrap,
                             [](const APInt &N1, const APInt &N2) {
                               bool Overflow;
                               (void)N1.sshl_ov(N2, Overflow);
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

  EnumerateInterestingConstantRanges([](const ConstantRange &CR) {
    unsigned Bits = CR.getBitWidth();
    CmpInst::Predicate Pred;
    APInt RHS, Offset;
    CR.getEquivalentICmp(Pred, RHS, Offset);
    EnumerateAPInts(Bits, [&](const APInt &N) {
      bool Result = ICmpInst::compare(N + Offset, RHS, Pred);
      EXPECT_EQ(CR.contains(N), Result);
    });

    if (CR.getEquivalentICmp(Pred, RHS)) {
      EnumerateAPInts(Bits, [&](const APInt &N) {
        bool Result = ICmpInst::compare(N, RHS, Pred);
        EXPECT_EQ(CR.contains(N), Result);
      });
    }
  });
}

#define EXPECT_MAY_OVERFLOW(op) \
  EXPECT_EQ(ConstantRange::OverflowResult::MayOverflow, (op))
#define EXPECT_ALWAYS_OVERFLOWS_LOW(op) \
  EXPECT_EQ(ConstantRange::OverflowResult::AlwaysOverflowsLow, (op))
#define EXPECT_ALWAYS_OVERFLOWS_HIGH(op) \
  EXPECT_EQ(ConstantRange::OverflowResult::AlwaysOverflowsHigh, (op))
#define EXPECT_NEVER_OVERFLOWS(op) \
  EXPECT_EQ(ConstantRange::OverflowResult::NeverOverflows, (op))

TEST_F(ConstantRangeTest, UnsignedAddOverflow) {
  // Ill-defined - may overflow is a conservative result.
  EXPECT_MAY_OVERFLOW(Some.unsignedAddMayOverflow(Empty));
  EXPECT_MAY_OVERFLOW(Empty.unsignedAddMayOverflow(Some));

  // Never overflow despite one full/wrap set.
  ConstantRange Zero(APInt::getZero(16));
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
  EXPECT_ALWAYS_OVERFLOWS_HIGH(A.unsignedAddMayOverflow(C2));
  EXPECT_MAY_OVERFLOW(C1.unsignedAddMayOverflow(A));
  EXPECT_ALWAYS_OVERFLOWS_HIGH(C2.unsignedAddMayOverflow(A));
}

TEST_F(ConstantRangeTest, UnsignedSubOverflow) {
  // Ill-defined - may overflow is a conservative result.
  EXPECT_MAY_OVERFLOW(Some.unsignedSubMayOverflow(Empty));
  EXPECT_MAY_OVERFLOW(Empty.unsignedSubMayOverflow(Some));

  // Never overflow despite one full/wrap set.
  ConstantRange Zero(APInt::getZero(16));
  ConstantRange Max(APInt::getAllOnes(16));
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
  EXPECT_ALWAYS_OVERFLOWS_LOW(A.unsignedSubMayOverflow(B));

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
  ConstantRange Zero(APInt::getZero(16));
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
  EXPECT_ALWAYS_OVERFLOWS_HIGH(A.signedAddMayOverflow(B6));

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
  EXPECT_ALWAYS_OVERFLOWS_LOW(C.signedAddMayOverflow(D6));

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
  ConstantRange Zero(APInt::getZero(16));
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
  EXPECT_ALWAYS_OVERFLOWS_HIGH(A.signedSubMayOverflow(B4));

  ConstantRange C(APInt(16, 0x8200), APInt(16, 0x8300));
  ConstantRange D1(APInt(16, 0x0100), APInt(16, 0x0201));
  ConstantRange D2(APInt(16, 0x0100), APInt(16, 0x0202));
  EXPECT_NEVER_OVERFLOWS(C.signedSubMayOverflow(D1));
  EXPECT_MAY_OVERFLOW(C.signedSubMayOverflow(D2));
  ConstantRange D3(APInt(16, 0x0299), APInt(16, 0x0400));
  ConstantRange D4(APInt(16, 0x0300), APInt(16, 0x0400));
  EXPECT_MAY_OVERFLOW(C.signedSubMayOverflow(D3));
  EXPECT_ALWAYS_OVERFLOWS_LOW(C.signedSubMayOverflow(D4));

  ConstantRange E(APInt(16, 0xff00), APInt(16, 0x0100));
  EXPECT_NEVER_OVERFLOWS(E.signedSubMayOverflow(E));
  ConstantRange F(APInt(16, 0xf000), APInt(16, 0x7001));
  EXPECT_MAY_OVERFLOW(F.signedSubMayOverflow(F));
}

template <typename Fn1, typename Fn2>
static void TestOverflowExhaustive(Fn1 OverflowFn, Fn2 MayOverflowFn) {
  // Constant range overflow checks are tested exhaustively on 4-bit numbers.
  EnumerateTwoInterestingConstantRanges([=](const ConstantRange &CR1,
                                            const ConstantRange &CR2) {
    // Loop over all N1 in CR1 and N2 in CR2 and check whether any of the
    // operations have overflow / have no overflow.
    bool RangeHasOverflowLow = false;
    bool RangeHasOverflowHigh = false;
    bool RangeHasNoOverflow = false;
    ForeachNumInConstantRange(CR1, [&](const APInt &N1) {
      ForeachNumInConstantRange(CR2, [&](const APInt &N2) {
        bool IsOverflowHigh;
        if (!OverflowFn(IsOverflowHigh, N1, N2)) {
          RangeHasNoOverflow = true;
          return;
        }

        if (IsOverflowHigh)
          RangeHasOverflowHigh = true;
        else
          RangeHasOverflowLow = true;
      });
    });

    ConstantRange::OverflowResult OR = MayOverflowFn(CR1, CR2);
    switch (OR) {
    case ConstantRange::OverflowResult::AlwaysOverflowsLow:
      EXPECT_TRUE(RangeHasOverflowLow);
      EXPECT_FALSE(RangeHasOverflowHigh);
      EXPECT_FALSE(RangeHasNoOverflow);
      break;
    case ConstantRange::OverflowResult::AlwaysOverflowsHigh:
      EXPECT_TRUE(RangeHasOverflowHigh);
      EXPECT_FALSE(RangeHasOverflowLow);
      EXPECT_FALSE(RangeHasNoOverflow);
      break;
    case ConstantRange::OverflowResult::NeverOverflows:
      EXPECT_FALSE(RangeHasOverflowLow);
      EXPECT_FALSE(RangeHasOverflowHigh);
      EXPECT_TRUE(RangeHasNoOverflow);
      break;
    case ConstantRange::OverflowResult::MayOverflow:
      // We return MayOverflow for empty sets as a conservative result,
      // but of course neither the RangeHasOverflow nor the
      // RangeHasNoOverflow flags will be set.
      if (CR1.isEmptySet() || CR2.isEmptySet())
        break;

      EXPECT_TRUE(RangeHasOverflowLow || RangeHasOverflowHigh);
      EXPECT_TRUE(RangeHasNoOverflow);
      break;
    }
  });
}

TEST_F(ConstantRangeTest, UnsignedAddOverflowExhaustive) {
  TestOverflowExhaustive(
      [](bool &IsOverflowHigh, const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.uadd_ov(N2, Overflow);
        IsOverflowHigh = true;
        return Overflow;
      },
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.unsignedAddMayOverflow(CR2);
      });
}

TEST_F(ConstantRangeTest, UnsignedSubOverflowExhaustive) {
  TestOverflowExhaustive(
      [](bool &IsOverflowHigh, const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.usub_ov(N2, Overflow);
        IsOverflowHigh = false;
        return Overflow;
      },
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.unsignedSubMayOverflow(CR2);
      });
}

TEST_F(ConstantRangeTest, UnsignedMulOverflowExhaustive) {
  TestOverflowExhaustive(
      [](bool &IsOverflowHigh, const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.umul_ov(N2, Overflow);
        IsOverflowHigh = true;
        return Overflow;
      },
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.unsignedMulMayOverflow(CR2);
      });
}

TEST_F(ConstantRangeTest, SignedAddOverflowExhaustive) {
  TestOverflowExhaustive(
      [](bool &IsOverflowHigh, const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.sadd_ov(N2, Overflow);
        IsOverflowHigh = N1.isNonNegative();
        return Overflow;
      },
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.signedAddMayOverflow(CR2);
      });
}

TEST_F(ConstantRangeTest, SignedSubOverflowExhaustive) {
  TestOverflowExhaustive(
      [](bool &IsOverflowHigh, const APInt &N1, const APInt &N2) {
        bool Overflow;
        (void) N1.ssub_ov(N2, Overflow);
        IsOverflowHigh = N1.isNonNegative();
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

      SmallBitVector Elems(1 << Bits);
      for (unsigned N = 0; N < Max; ++N) {
        APInt Num(Bits, N);
        if ((Num & Known.Zero) != 0 || (~Num & Known.One) != 0)
          continue;
        Elems.set(Num.getZExtValue());
      }

      TestRange(ConstantRange::fromKnownBits(Known, false),
                Elems, PreferSmallestUnsigned, {});
      TestRange(ConstantRange::fromKnownBits(Known, true),
                Elems, PreferSmallestSigned, {});
    }
  }
}

TEST_F(ConstantRangeTest, ToKnownBits) {
  EnumerateInterestingConstantRanges([&](const ConstantRange &CR) {
    KnownBits Known = CR.toKnownBits();
    KnownBits ExpectedKnown(CR.getBitWidth());
    ExpectedKnown.Zero.setAllBits();
    ExpectedKnown.One.setAllBits();
    ForeachNumInConstantRange(CR, [&](const APInt &N) {
      ExpectedKnown.One &= N;
      ExpectedKnown.Zero &= ~N;
    });
    // For an empty CR any result would be legal.
    if (!CR.isEmptySet()) {
      EXPECT_EQ(ExpectedKnown, Known);
    }
  });
}

TEST_F(ConstantRangeTest, Negative) {
  // All elements in an empty set (of which there are none) are both negative
  // and non-negative. Empty & full sets checked explicitly for clarity, but
  // they are also covered by the exhaustive test below.
  EXPECT_TRUE(Empty.isAllNegative());
  EXPECT_TRUE(Empty.isAllNonNegative());
  EXPECT_FALSE(Full.isAllNegative());
  EXPECT_FALSE(Full.isAllNonNegative());

  EnumerateInterestingConstantRanges([](const ConstantRange &CR) {
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

TEST_F(ConstantRangeTest, UAddSat) {
  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.uadd_sat(CR2);
      },
      [](const APInt &N1, const APInt &N2) {
        return N1.uadd_sat(N2);
      },
      PreferSmallestUnsigned);
}

TEST_F(ConstantRangeTest, USubSat) {
  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.usub_sat(CR2);
      },
      [](const APInt &N1, const APInt &N2) {
        return N1.usub_sat(N2);
      },
      PreferSmallestUnsigned);
}

TEST_F(ConstantRangeTest, UMulSat) {
  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.umul_sat(CR2);
      },
      [](const APInt &N1, const APInt &N2) { return N1.umul_sat(N2); },
      PreferSmallestUnsigned);
}

TEST_F(ConstantRangeTest, UShlSat) {
  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.ushl_sat(CR2);
      },
      [](const APInt &N1, const APInt &N2) { return N1.ushl_sat(N2); },
      PreferSmallestUnsigned);
}

TEST_F(ConstantRangeTest, SAddSat) {
  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.sadd_sat(CR2);
      },
      [](const APInt &N1, const APInt &N2) {
        return N1.sadd_sat(N2);
      },
      PreferSmallestSigned);
}

TEST_F(ConstantRangeTest, SSubSat) {
  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.ssub_sat(CR2);
      },
      [](const APInt &N1, const APInt &N2) {
        return N1.ssub_sat(N2);
      },
      PreferSmallestSigned);
}

TEST_F(ConstantRangeTest, SMulSat) {
  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.smul_sat(CR2);
      },
      [](const APInt &N1, const APInt &N2) { return N1.smul_sat(N2); },
      PreferSmallestSigned);
}

TEST_F(ConstantRangeTest, SShlSat) {
  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.sshl_sat(CR2);
      },
      [](const APInt &N1, const APInt &N2) { return N1.sshl_sat(N2); },
      PreferSmallestSigned);
}

TEST_F(ConstantRangeTest, Abs) {
  TestUnaryOpExhaustive(
      [](const ConstantRange &CR) { return CR.abs(); },
      [](const APInt &N) { return N.abs(); });

  TestUnaryOpExhaustive(
      [](const ConstantRange &CR) { return CR.abs(/*IntMinIsPoison=*/true); },
      [](const APInt &N) -> std::optional<APInt> {
        if (N.isMinSignedValue())
          return std::nullopt;
        return N.abs();
      });
}

TEST_F(ConstantRangeTest, Ctlz) {
  TestUnaryOpExhaustive(
      [](const ConstantRange &CR) { return CR.ctlz(); },
      [](const APInt &N) { return APInt(N.getBitWidth(), N.countl_zero()); });

  TestUnaryOpExhaustive(
      [](const ConstantRange &CR) { return CR.ctlz(/*ZeroIsPoison=*/true); },
      [](const APInt &N) -> std::optional<APInt> {
        if (N.isZero())
          return std::nullopt;
        return APInt(N.getBitWidth(), N.countl_zero());
      });
}

TEST_F(ConstantRangeTest, Cttz) {
  TestUnaryOpExhaustive(
      [](const ConstantRange &CR) { return CR.cttz(); },
      [](const APInt &N) { return APInt(N.getBitWidth(), N.countr_zero()); });

  TestUnaryOpExhaustive(
      [](const ConstantRange &CR) { return CR.cttz(/*ZeroIsPoison=*/true); },
      [](const APInt &N) -> std::optional<APInt> {
        if (N.isZero())
          return std::nullopt;
        return APInt(N.getBitWidth(), N.countr_zero());
      });
}

TEST_F(ConstantRangeTest, Ctpop) {
  TestUnaryOpExhaustive(
      [](const ConstantRange &CR) { return CR.ctpop(); },
      [](const APInt &N) { return APInt(N.getBitWidth(), N.popcount()); });
}

TEST_F(ConstantRangeTest, castOps) {
  ConstantRange A(APInt(16, 66), APInt(16, 128));
  ConstantRange FpToI8 = A.castOp(Instruction::FPToSI, 8);
  EXPECT_EQ(8u, FpToI8.getBitWidth());
  EXPECT_TRUE(FpToI8.isFullSet());

  ConstantRange FpToI16 = A.castOp(Instruction::FPToSI, 16);
  EXPECT_EQ(16u, FpToI16.getBitWidth());
  EXPECT_EQ(A, FpToI16);

  ConstantRange FPExtToDouble = A.castOp(Instruction::FPExt, 64);
  EXPECT_EQ(64u, FPExtToDouble.getBitWidth());
  EXPECT_TRUE(FPExtToDouble.isFullSet());

  ConstantRange PtrToInt = A.castOp(Instruction::PtrToInt, 64);
  EXPECT_EQ(64u, PtrToInt.getBitWidth());
  EXPECT_TRUE(PtrToInt.isFullSet());

  ConstantRange IntToPtr = A.castOp(Instruction::IntToPtr, 64);
  EXPECT_EQ(64u, IntToPtr.getBitWidth());
  EXPECT_TRUE(IntToPtr.isFullSet());

  ConstantRange UIToFP = A.castOp(Instruction::UIToFP, 16);
  EXPECT_EQ(16u, UIToFP.getBitWidth());
  EXPECT_TRUE(UIToFP.isFullSet());

  ConstantRange UIToFP2 = A.castOp(Instruction::UIToFP, 64);
  ConstantRange B(APInt(64, 0), APInt(64, 65536));
  EXPECT_EQ(64u, UIToFP2.getBitWidth());
  EXPECT_EQ(B, UIToFP2);

  ConstantRange SIToFP = A.castOp(Instruction::SIToFP, 16);
  EXPECT_EQ(16u, SIToFP.getBitWidth());
  EXPECT_TRUE(SIToFP.isFullSet());

  ConstantRange SIToFP2 = A.castOp(Instruction::SIToFP, 64);
  ConstantRange C(APInt(64, -32768), APInt(64, 32768));
  EXPECT_EQ(64u, SIToFP2.getBitWidth());
  EXPECT_EQ(C, SIToFP2);
}

TEST_F(ConstantRangeTest, binaryAnd) {
  // Single element ranges.
  ConstantRange R16(APInt(8, 16));
  ConstantRange R20(APInt(8, 20));
  EXPECT_EQ(*R16.binaryAnd(R16).getSingleElement(), APInt(8, 16));
  EXPECT_EQ(*R16.binaryAnd(R20).getSingleElement(), APInt(8, 16 & 20));

  ConstantRange R16_32(APInt(8, 16), APInt(8, 32));
  // 'And' with a high bits mask.
  ConstantRange R32(APInt(8, 32));
  EXPECT_TRUE(R16_32.binaryAnd(R32).getSingleElement()->isZero());
  EXPECT_TRUE(R32.binaryAnd(R16_32).getSingleElement()->isZero());
  // 'And' with a low bits mask. Handled conservatively for now.
  ConstantRange R4(APInt(8, 4));
  ConstantRange R0_5(APInt(8, 0), APInt(8, 5));
  EXPECT_EQ(R16_32.binaryAnd(R4), R0_5);
  EXPECT_EQ(R4.binaryAnd(R16_32), R0_5);

  // Ranges with more than one element. Handled conservatively for now.
  ConstantRange R0_99(APInt(8, 0), APInt(8, 99));
  ConstantRange R0_32(APInt(8, 0), APInt(8, 32));
  EXPECT_EQ(R16_32.binaryAnd(R0_99), R0_32);
  EXPECT_EQ(R0_99.binaryAnd(R16_32), R0_32);

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.binaryAnd(CR2);
      },
      [](const APInt &N1, const APInt &N2) { return N1 & N2; }, PreferSmallest,
      CheckSingleElementsOnly);
}

TEST_F(ConstantRangeTest, binaryOr) {
  // Single element ranges.
  ConstantRange R16(APInt(8, 16));
  ConstantRange R20(APInt(8, 20));
  EXPECT_EQ(*R16.binaryOr(R16).getSingleElement(), APInt(8, 16));
  EXPECT_EQ(*R16.binaryOr(R20).getSingleElement(), APInt(8, 16 | 20));

  ConstantRange R16_32(APInt(8, 16), APInt(8, 32));
  // 'Or' with a high bits mask.
  // KnownBits estimate is important, otherwise the maximum included element
  // would be 2^8 - 1.
  ConstantRange R32(APInt(8, 32));
  ConstantRange R48_64(APInt(8, 48), APInt(8, 64));
  EXPECT_EQ(R16_32.binaryOr(R32), R48_64);
  EXPECT_EQ(R32.binaryOr(R16_32), R48_64);
  // 'Or' with a low bits mask.
  ConstantRange R4(APInt(8, 4));
  ConstantRange R0_16(APInt(8, 0), APInt(8, 16));
  ConstantRange R4_16(APInt(8, 4), APInt(8, 16));
  EXPECT_EQ(R0_16.binaryOr(R4), R4_16);
  EXPECT_EQ(R4.binaryOr(R0_16), R4_16);

  // Ranges with more than one element. Handled conservatively for now.
  // UMaxUMin estimate is important, otherwise the lower bound would be zero.
  ConstantRange R0_64(APInt(8, 0), APInt(8, 64));
  ConstantRange R5_32(APInt(8, 5), APInt(8, 32));
  ConstantRange R5_64(APInt(8, 5), APInt(8, 64));
  EXPECT_EQ(R0_64.binaryOr(R5_32), R5_64);
  EXPECT_EQ(R5_32.binaryOr(R0_64), R5_64);

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.binaryOr(CR2);
      },
      [](const APInt &N1, const APInt &N2) { return N1 | N2; }, PreferSmallest,
      CheckSingleElementsOnly);
}

TEST_F(ConstantRangeTest, binaryXor) {
  // Single element ranges.
  ConstantRange R16(APInt(8, 16));
  ConstantRange R20(APInt(8, 20));
  EXPECT_EQ(*R16.binaryXor(R16).getSingleElement(), APInt(8, 0));
  EXPECT_EQ(*R16.binaryXor(R20).getSingleElement(), APInt(8, 16 ^ 20));

  // Ranges with more than a single element.
  ConstantRange R16_35(APInt(8, 16), APInt(8, 35));
  ConstantRange R0_99(APInt(8, 0), APInt(8, 99));
  EXPECT_EQ(R16_35.binaryXor(R16_35), ConstantRange(APInt(8, 0), APInt(8, 64)));
  EXPECT_EQ(R16_35.binaryXor(R0_99), ConstantRange(APInt(8, 0), APInt(8, 128)));
  EXPECT_EQ(R0_99.binaryXor(R16_35), ConstantRange(APInt(8, 0), APInt(8, 128)));

  // Treat xor A, B as sub nsw nuw A, B
  ConstantRange R0_51(APInt(8, 0), APInt(8, 51));
  ConstantRange R63(APInt(8, 63));
  EXPECT_EQ(R0_51.binaryXor(R63), ConstantRange(APInt(8, 13), APInt(8, 64)));
  EXPECT_EQ(R63.binaryXor(R0_51), ConstantRange(APInt(8, 13), APInt(8, 64)));

  TestBinaryOpExhaustive(
      [](const ConstantRange &CR1, const ConstantRange &CR2) {
        return CR1.binaryXor(CR2);
      },
      [](const APInt &N1, const APInt &N2) {
        return N1 ^ N2;
      },
      PreferSmallest,
      CheckSingleElementsOnly);
}

TEST_F(ConstantRangeTest, binaryNot) {
  TestUnaryOpExhaustive(
      [](const ConstantRange &CR) { return CR.binaryNot(); },
      [](const APInt &N) { return ~N; },
      PreferSmallest);
  TestUnaryOpExhaustive(
      [](const ConstantRange &CR) {
        return CR.binaryXor(ConstantRange(APInt::getAllOnes(CR.getBitWidth())));
      },
      [](const APInt &N) { return ~N; }, PreferSmallest);
  TestUnaryOpExhaustive(
      [](const ConstantRange &CR) {
        return ConstantRange(APInt::getAllOnes(CR.getBitWidth())).binaryXor(CR);
      },
      [](const APInt &N) { return ~N; }, PreferSmallest);
}

template <typename T>
void testConstantRangeICmpPredEquivalence(ICmpInst::Predicate SrcPred, T Func) {
  EnumerateTwoInterestingConstantRanges(
      [&](const ConstantRange &CR1, const ConstantRange &CR2) {
        ICmpInst::Predicate TgtPred;
        bool ExpectedEquivalent;
        std::tie(TgtPred, ExpectedEquivalent) = Func(CR1, CR2);
        if (TgtPred == CmpInst::Predicate::BAD_ICMP_PREDICATE)
          return;
        bool TrulyEquivalent = true;
        ForeachNumInConstantRange(CR1, [&](const APInt &N1) {
          if (!TrulyEquivalent)
            return;
          ForeachNumInConstantRange(CR2, [&](const APInt &N2) {
            if (!TrulyEquivalent)
              return;
            TrulyEquivalent &= ICmpInst::compare(N1, N2, SrcPred) ==
                               ICmpInst::compare(N1, N2, TgtPred);
          });
        });
        ASSERT_EQ(TrulyEquivalent, ExpectedEquivalent);
      });
}

TEST_F(ConstantRangeTest, areInsensitiveToSignednessOfICmpPredicate) {
  for (auto Pred : ICmpInst::predicates()) {
    if (ICmpInst::isEquality(Pred))
      continue;
    ICmpInst::Predicate FlippedSignednessPred =
        ICmpInst::getFlippedSignednessPredicate(Pred);
    testConstantRangeICmpPredEquivalence(Pred, [FlippedSignednessPred](
                                                   const ConstantRange &CR1,
                                                   const ConstantRange &CR2) {
      return std::make_pair(
          FlippedSignednessPred,
          ConstantRange::areInsensitiveToSignednessOfICmpPredicate(CR1, CR2));
    });
  }
}

TEST_F(ConstantRangeTest, areInsensitiveToSignednessOfInvertedICmpPredicate) {
  for (auto Pred : ICmpInst::predicates()) {
    if (ICmpInst::isEquality(Pred))
      continue;
    ICmpInst::Predicate InvertedFlippedSignednessPred =
        ICmpInst::getInversePredicate(
            ICmpInst::getFlippedSignednessPredicate(Pred));
    testConstantRangeICmpPredEquivalence(
        Pred, [InvertedFlippedSignednessPred](const ConstantRange &CR1,
                                              const ConstantRange &CR2) {
          return std::make_pair(
              InvertedFlippedSignednessPred,
              ConstantRange::areInsensitiveToSignednessOfInvertedICmpPredicate(
                  CR1, CR2));
        });
  }
}

TEST_F(ConstantRangeTest, getEquivalentPredWithFlippedSignedness) {
  for (auto Pred : ICmpInst::predicates()) {
    if (ICmpInst::isEquality(Pred))
      continue;
    testConstantRangeICmpPredEquivalence(
        Pred, [Pred](const ConstantRange &CR1, const ConstantRange &CR2) {
          return std::make_pair(
              ConstantRange::getEquivalentPredWithFlippedSignedness(Pred, CR1,
                                                                    CR2),
              /*ExpectedEquivalent=*/true);
        });
  }
}

TEST_F(ConstantRangeTest, isSizeLargerThan) {
  EXPECT_FALSE(Empty.isSizeLargerThan(0));

  EXPECT_TRUE(Full.isSizeLargerThan(0));
  EXPECT_TRUE(Full.isSizeLargerThan(65535));
  EXPECT_FALSE(Full.isSizeLargerThan(65536));

  EXPECT_TRUE(One.isSizeLargerThan(0));
  EXPECT_FALSE(One.isSizeLargerThan(1));
}

TEST_F(ConstantRangeTest, MakeMaskNotEqualRange) {
  // Mask: 0b0001, C: 0b0001. MMNE() = [2, 1)
  ConstantRange CR(APInt(4, 2), APInt(4, 1));
  EXPECT_EQ(CR, ConstantRange::makeMaskNotEqualRange(APInt(4, 1), APInt(4, 1)));
  EXPECT_NE(CR, ConstantRange::makeMaskNotEqualRange(APInt(4, 0),
                                                     APInt(4, -1, true)));
  EXPECT_TRUE(CR.contains(APInt(4, 7)));
  EXPECT_TRUE(CR.contains(APInt(4, 15)));

  // Mask: 0b0100, C: 0b0100. MMNE() = [-8, 4)
  ConstantRange CR2(APInt(4, -8, true), APInt(4, 4));
  auto MMNE = ConstantRange::makeMaskNotEqualRange(APInt(4, 4), APInt(4, 4));
  EXPECT_EQ(CR2, MMNE);
  EXPECT_NE(ConstantRange::getNonEmpty(APInt(4, 0), APInt(4, -4, true)), MMNE);

  // CR: [-16, -8). MMNE() = [-8, -16)
  ConstantRange CR3(APInt(8, 240), APInt(8, 248));
  EXPECT_EQ(CR3.inverse(),
            ConstantRange::makeMaskNotEqualRange(APInt(8, 248), APInt(8, 240)));

  // Mask: 0, C: 0b1111: unsatisfiable.
  EXPECT_EQ(ConstantRange::getFull(4),
            ConstantRange::makeMaskNotEqualRange(APInt(4, 0), APInt(4, 15)));
}

TEST_F(ConstantRangeTest, MakeMaskNotEqualRangeExhaustive) {
  unsigned Bits = 4;
  unsigned Max = 1 << Bits;

  EnumerateAPInts(Bits, [&](const APInt &Mask) {
    EnumerateAPInts(Bits, [&](const APInt &C) {
      SmallBitVector Elems(Max);
      for (unsigned N = 0; N < Max; ++N) {
        APInt Num(Bits, N);
        if ((Num & Mask) == C)
          continue;
        Elems.set(Num.getZExtValue());
      }

      // Only test optimality with PreferSmallest. E.g., given Mask = 0b0001, C
      // = 0b0001, a possible better range would be [0, 15) when preferring the
      // smallest unsigned, however we conservatively return [2, 1).
      TestRange(ConstantRange::makeMaskNotEqualRange(Mask, C), Elems,
                PreferSmallest, {});
    });
  });
}

} // anonymous namespace
