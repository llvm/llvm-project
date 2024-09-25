//===- ConstantRangeTest.cpp - ConstantRange tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ConstantFPRange.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/KnownBits.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class ConstantFPRangeTest : public ::testing::Test {
protected:
  static const fltSemantics &Sem;
  static ConstantFPRange Full;
  static ConstantFPRange Empty;
  static ConstantFPRange Finite;
  static ConstantFPRange One;
  static ConstantFPRange PosZero;
  static ConstantFPRange NegZero;
  static ConstantFPRange Zero;
  static ConstantFPRange PosInf;
  static ConstantFPRange NegInf;
  static ConstantFPRange Denormal;
  static ConstantFPRange NaN;
  static ConstantFPRange SNaN;
  static ConstantFPRange QNaN;
  static ConstantFPRange Some;
  static ConstantFPRange SomePos;
  static ConstantFPRange SomeNeg;
};

const fltSemantics &ConstantFPRangeTest::Sem = APFloat::IEEEdouble();
ConstantFPRange ConstantFPRangeTest::Full =
    ConstantFPRange::getFull(APFloat::IEEEdouble());
ConstantFPRange ConstantFPRangeTest::Empty =
    ConstantFPRange::getEmpty(APFloat::IEEEdouble());
ConstantFPRange ConstantFPRangeTest::Finite =
    ConstantFPRange::getFinite(APFloat::IEEEdouble());
ConstantFPRange ConstantFPRangeTest::One = ConstantFPRange(APFloat(1.0));
ConstantFPRange ConstantFPRangeTest::PosZero = ConstantFPRange(
    APFloat::getZero(APFloat::IEEEdouble(), /*Negative=*/false));
ConstantFPRange ConstantFPRangeTest::NegZero =
    ConstantFPRange(APFloat::getZero(APFloat::IEEEdouble(), /*Negative=*/true));
ConstantFPRange ConstantFPRangeTest::Zero = ConstantFPRange::getNonNaN(
    APFloat::getZero(APFloat::IEEEdouble(), /*Negative=*/true),
    APFloat::getZero(APFloat::IEEEdouble(), /*Negative=*/false));
ConstantFPRange ConstantFPRangeTest::Denormal =
    ConstantFPRange(APFloat::getSmallest(APFloat::IEEEdouble()));
ConstantFPRange ConstantFPRangeTest::PosInf =
    ConstantFPRange(APFloat::getInf(APFloat::IEEEdouble(), /*Negative=*/false));
ConstantFPRange ConstantFPRangeTest::NegInf =
    ConstantFPRange(APFloat::getInf(APFloat::IEEEdouble(), /*Negative=*/true));
ConstantFPRange ConstantFPRangeTest::NaN = ConstantFPRange::getNaNOnly(
    APFloat::IEEEdouble(), /*MayBeQNaN=*/true, /*MayBeSNaN=*/true);
ConstantFPRange ConstantFPRangeTest::SNaN =
    ConstantFPRange(APFloat::getSNaN(APFloat::IEEEdouble()));
ConstantFPRange ConstantFPRangeTest::QNaN =
    ConstantFPRange(APFloat::getQNaN(APFloat::IEEEdouble()));
ConstantFPRange ConstantFPRangeTest::Some =
    ConstantFPRange::getNonNaN(APFloat(-3.0), APFloat(3.0));
ConstantFPRange ConstantFPRangeTest::SomePos = ConstantFPRange::getNonNaN(
    APFloat::getZero(APFloat::IEEEdouble(), /*Negative=*/false), APFloat(3.0));
ConstantFPRange ConstantFPRangeTest::SomeNeg = ConstantFPRange::getNonNaN(
    APFloat(-3.0), APFloat::getZero(APFloat::IEEEdouble(), /*Negative=*/true));

static void strictNext(APFloat &V) {
  // Note: nextUp(+/-0) is smallest.
  if (V.isNegZero())
    V = APFloat::getZero(V.getSemantics(), /*Negative=*/false);
  else
    V.next(/*nextDown=*/false);
}

template <typename Fn>
static void EnumerateConstantFPRangesImpl(Fn TestFn, bool Exhaustive,
                                          bool MayBeQNaN, bool MayBeSNaN) {
  const fltSemantics &Sem = APFloat::Float8E4M3();
  APFloat PosInf = APFloat::getInf(Sem, /*Negative=*/false);
  APFloat NegInf = APFloat::getInf(Sem, /*Negative=*/true);
  TestFn(ConstantFPRange(PosInf, NegInf, MayBeQNaN, MayBeSNaN));

  if (!Exhaustive) {
    SmallVector<APFloat, 36> Values;
    Values.push_back(APFloat::getInf(Sem, /*Negative=*/true));
    Values.push_back(APFloat::getLargest(Sem, /*Negative=*/true));
    unsigned BitWidth = APFloat::semanticsSizeInBits(Sem);
    unsigned Exponents = APFloat::semanticsMaxExponent(Sem) -
                         APFloat::semanticsMinExponent(Sem) + 3;
    unsigned MantissaBits = APFloat::semanticsPrecision(Sem) - 1;
    // Add -2^(max exponent), -2^(max exponent-1), ..., -2^(min exponent)
    for (unsigned M = Exponents - 2; M != 0; --M)
      Values.push_back(
          APFloat(Sem, APInt(BitWidth, (M + Exponents) << MantissaBits)));
    Values.push_back(APFloat::getSmallest(Sem, /*Negative=*/true));
    Values.push_back(APFloat::getZero(Sem, /*Negative=*/true));
    size_t E = Values.size();
    for (size_t I = 1; I <= E; ++I)
      Values.push_back(-Values[E - I]);
    for (size_t I = 0; I != Values.size(); ++I)
      for (size_t J = I; J != Values.size(); ++J)
        TestFn(ConstantFPRange(Values[I], Values[J], MayBeQNaN, MayBeSNaN));
    return;
  }

  auto Next = [&](APFloat &V) {
    if (V.isPosInfinity())
      return false;
    strictNext(V);
    return true;
  };

  APFloat Lower = NegInf;
  do {
    APFloat Upper = Lower;
    do {
      TestFn(ConstantFPRange(Lower, Upper, MayBeQNaN, MayBeSNaN));
    } while (Next(Upper));
  } while (Next(Lower));
}

template <typename Fn>
static void EnumerateConstantFPRanges(Fn TestFn, bool Exhaustive) {
  EnumerateConstantFPRangesImpl(TestFn, Exhaustive, /*MayBeQNaN=*/false,
                                /*MayBeSNaN=*/false);
  EnumerateConstantFPRangesImpl(TestFn, Exhaustive, /*MayBeQNaN=*/false,
                                /*MayBeSNaN=*/true);
  EnumerateConstantFPRangesImpl(TestFn, Exhaustive, /*MayBeQNaN=*/true,
                                /*MayBeSNaN=*/false);
  EnumerateConstantFPRangesImpl(TestFn, Exhaustive, /*MayBeQNaN=*/true,
                                /*MayBeSNaN=*/true);
}

template <typename Fn>
static void EnumerateTwoInterestingConstantFPRanges(Fn TestFn,
                                                    bool Exhaustive) {
  EnumerateConstantFPRanges(
      [&](const ConstantFPRange &CR1) {
        EnumerateConstantFPRanges(
            [&](const ConstantFPRange &CR2) { TestFn(CR1, CR2); }, Exhaustive);
      },
      Exhaustive);
}

template <typename Fn>
static void EnumerateValuesInConstantFPRange(const ConstantFPRange &CR,
                                             Fn TestFn) {
  const fltSemantics &Sem = CR.getSemantics();
  unsigned Bits = APFloat::semanticsSizeInBits(Sem);
  assert(Bits < 32 && "Too many bits");
  for (unsigned I = 0, E = (1U << Bits) - 1; I != E; ++I) {
    APFloat V(Sem, APInt(Bits, I));
    if (CR.contains(V))
      TestFn(V);
  }
}

TEST_F(ConstantFPRangeTest, Basics) {
  EXPECT_TRUE(Full.isFullSet());
  EXPECT_FALSE(Full.isEmptySet());
  EXPECT_TRUE(Full.contains(APFloat::getNaN(Sem)));
  EXPECT_TRUE(Full.contains(APFloat::getInf(Sem, /*Negative=*/false)));
  EXPECT_TRUE(Full.contains(APFloat::getInf(Sem, /*Negative=*/true)));
  EXPECT_TRUE(Full.contains(APFloat::getZero(Sem, /*Negative=*/false)));
  EXPECT_TRUE(Full.contains(APFloat::getZero(Sem, /*Negative=*/true)));
  EXPECT_TRUE(Full.contains(APFloat::getSmallest(Sem)));
  EXPECT_TRUE(Full.contains(APFloat(2.0)));
  EXPECT_TRUE(Full.contains(Full));
  EXPECT_TRUE(Full.contains(Empty));
  EXPECT_TRUE(Full.contains(Finite));
  EXPECT_TRUE(Full.contains(Zero));
  EXPECT_TRUE(Full.contains(Some));

  EXPECT_FALSE(Empty.isFullSet());
  EXPECT_TRUE(Empty.isEmptySet());
  EXPECT_FALSE(Empty.contains(APFloat::getNaN(Sem)));
  EXPECT_FALSE(Empty.contains(APFloat::getInf(Sem, /*Negative=*/false)));
  EXPECT_FALSE(Empty.contains(APFloat::getZero(Sem, /*Negative=*/true)));
  EXPECT_FALSE(Empty.contains(APFloat(2.0)));
  EXPECT_TRUE(Empty.contains(Empty));

  EXPECT_FALSE(Finite.isFullSet());
  EXPECT_FALSE(Finite.isEmptySet());
  EXPECT_FALSE(Finite.contains(APFloat::getNaN(Sem)));
  EXPECT_FALSE(Finite.contains(APFloat::getInf(Sem, /*Negative=*/false)));
  EXPECT_FALSE(Finite.contains(APFloat::getInf(Sem, /*Negative=*/true)));
  EXPECT_TRUE(Finite.contains(APFloat::getLargest(Sem, /*Negative=*/false)));
  EXPECT_TRUE(Finite.contains(APFloat::getLargest(Sem, /*Negative=*/true)));
  EXPECT_TRUE(Finite.contains(Finite));
  EXPECT_TRUE(Finite.contains(Some));
  EXPECT_TRUE(Finite.contains(Denormal));
  EXPECT_TRUE(Finite.contains(Zero));
  EXPECT_FALSE(Finite.contains(PosInf));
  EXPECT_FALSE(Finite.contains(NaN));

  EXPECT_TRUE(One.contains(APFloat(1.0)));
  EXPECT_FALSE(One.contains(APFloat(1.1)));

  EXPECT_TRUE(PosZero.contains(APFloat::getZero(Sem, /*Negative=*/false)));
  EXPECT_FALSE(PosZero.contains(APFloat::getZero(Sem, /*Negative=*/true)));
  EXPECT_TRUE(NegZero.contains(APFloat::getZero(Sem, /*Negative=*/true)));
  EXPECT_FALSE(NegZero.contains(APFloat::getZero(Sem, /*Negative=*/false)));
  EXPECT_TRUE(Zero.contains(PosZero));
  EXPECT_TRUE(Zero.contains(NegZero));
  EXPECT_TRUE(Denormal.contains(APFloat::getSmallest(Sem)));
  EXPECT_FALSE(Denormal.contains(APFloat::getSmallestNormalized(Sem)));
  EXPECT_TRUE(PosInf.contains(APFloat::getInf(Sem, /*Negative=*/false)));
  EXPECT_TRUE(NegInf.contains(APFloat::getInf(Sem, /*Negative=*/true)));
  EXPECT_TRUE(NaN.contains(APFloat::getQNaN(Sem)));
  EXPECT_TRUE(NaN.contains(APFloat::getSNaN(Sem)));
  EXPECT_TRUE(NaN.contains(SNaN));
  EXPECT_TRUE(NaN.contains(QNaN));

  EXPECT_TRUE(Some.contains(APFloat(3.0)));
  EXPECT_TRUE(Some.contains(APFloat(-3.0)));
  EXPECT_FALSE(Some.contains(APFloat(4.0)));
  APFloat Next1(3.0);
  Next1.next(/*nextDown=*/true);
  EXPECT_TRUE(Some.contains(Next1));
  APFloat Next2(3.0);
  Next2.next(/*nextDown=*/false);
  EXPECT_FALSE(Some.contains(Next2));
  EXPECT_TRUE(Some.contains(Zero));
  EXPECT_TRUE(Some.contains(Some));
  EXPECT_TRUE(Some.contains(One));
  EXPECT_FALSE(Some.contains(NaN));
  EXPECT_FALSE(Some.contains(PosInf));
  EXPECT_TRUE(SomePos.contains(APFloat(3.0)));
  EXPECT_FALSE(SomeNeg.contains(APFloat(3.0)));
  EXPECT_TRUE(SomeNeg.contains(APFloat(-3.0)));
  EXPECT_FALSE(SomePos.contains(APFloat(-3.0)));
  EXPECT_TRUE(Some.contains(SomePos));
  EXPECT_TRUE(Some.contains(SomeNeg));
}

TEST_F(ConstantFPRangeTest, Equality) {
  EXPECT_EQ(Full, Full);
  EXPECT_EQ(Empty, Empty);
  EXPECT_EQ(One, One);
  EXPECT_EQ(Some, Some);
  EXPECT_NE(Full, Empty);
  EXPECT_NE(Zero, PosZero);
  EXPECT_NE(One, NaN);
  EXPECT_NE(Some, One);
  EXPECT_NE(SNaN, QNaN);
}

TEST_F(ConstantFPRangeTest, SingleElement) {
  EXPECT_EQ(Full.getSingleElement(), static_cast<APFloat *>(nullptr));
  EXPECT_EQ(Empty.getSingleElement(), static_cast<APFloat *>(nullptr));
  EXPECT_EQ(Finite.getSingleElement(), static_cast<APFloat *>(nullptr));
  EXPECT_EQ(Zero.getSingleElement(), static_cast<APFloat *>(nullptr));
  EXPECT_EQ(NaN.getSingleElement(), static_cast<APFloat *>(nullptr));
  EXPECT_EQ(SNaN.getSingleElement(), static_cast<APFloat *>(nullptr));
  EXPECT_EQ(QNaN.getSingleElement(), static_cast<APFloat *>(nullptr));

  EXPECT_EQ(*One.getSingleElement(), APFloat(1.0));
  EXPECT_EQ(*PosZero.getSingleElement(), APFloat::getZero(Sem));
  EXPECT_EQ(*PosInf.getSingleElement(), APFloat::getInf(Sem));

  EXPECT_FALSE(Full.isSingleElement());
  EXPECT_FALSE(Empty.isSingleElement());
  EXPECT_TRUE(One.isSingleElement());
  EXPECT_FALSE(Some.isSingleElement());
  EXPECT_FALSE(Zero.isSingleElement());
}

TEST_F(ConstantFPRangeTest, ExhaustivelyEnumerate) {
  constexpr unsigned NNaNValues = (1 << 8) - 2 * ((1 << 3) - 1);
  constexpr unsigned Expected = 4 * ((NNaNValues + 1) * NNaNValues / 2 + 1);
  unsigned Count = 0;
  EnumerateConstantFPRanges([&](const ConstantFPRange &) { ++Count; },
                            /*Exhaustive=*/true);
  EXPECT_EQ(Expected, Count);
}

TEST_F(ConstantFPRangeTest, Enumerate) {
  constexpr unsigned NNaNValues = 2 * ((1 << 4) - 2 + 4);
  constexpr unsigned Expected = 4 * ((NNaNValues + 1) * NNaNValues / 2 + 1);
  unsigned Count = 0;
  EnumerateConstantFPRanges([&](const ConstantFPRange &) { ++Count; },
                            /*Exhaustive=*/false);
  EXPECT_EQ(Expected, Count);
}

TEST_F(ConstantFPRangeTest, IntersectWith) {
  EXPECT_EQ(Empty.intersectWith(Full), Empty);
  EXPECT_EQ(Empty.intersectWith(Empty), Empty);
  EXPECT_EQ(Empty.intersectWith(One), Empty);
  EXPECT_EQ(Empty.intersectWith(Some), Empty);
  EXPECT_EQ(Full.intersectWith(Full), Full);
  EXPECT_EQ(Some.intersectWith(Some), Some);
  EXPECT_EQ(Some.intersectWith(One), One);
  EXPECT_EQ(Full.intersectWith(One), One);
  EXPECT_EQ(Full.intersectWith(Some), Some);
  EXPECT_EQ(Some.intersectWith(SomePos), SomePos);
  EXPECT_EQ(Some.intersectWith(SomeNeg), SomeNeg);
  EXPECT_EQ(NaN.intersectWith(Finite), Empty);
  EXPECT_EQ(NaN.intersectWith(SNaN), SNaN);
  EXPECT_EQ(NaN.intersectWith(QNaN), QNaN);
  EXPECT_EQ(Finite.intersectWith(One), One);
  EXPECT_EQ(Some.intersectWith(Zero), Zero);
  EXPECT_EQ(ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(4.0))
                .intersectWith(
                    ConstantFPRange::getNonNaN(APFloat(3.0), APFloat(6.0))),
            ConstantFPRange::getNonNaN(APFloat(3.0), APFloat(4.0)));
  EXPECT_EQ(ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(2.0))
                .intersectWith(
                    ConstantFPRange::getNonNaN(APFloat(5.0), APFloat(6.0))),
            Empty);
}

TEST_F(ConstantFPRangeTest, UnionWith) {
  EXPECT_EQ(Empty.unionWith(Full), Full);
  EXPECT_EQ(Empty.unionWith(Empty), Empty);
  EXPECT_EQ(Empty.unionWith(One), One);
  EXPECT_EQ(Empty.unionWith(Some), Some);
  EXPECT_EQ(Full.unionWith(Full), Full);
  EXPECT_EQ(Some.unionWith(Some), Some);
  EXPECT_EQ(Some.unionWith(One), Some);
  EXPECT_EQ(Full.unionWith(Some), Full);
  EXPECT_EQ(Some.unionWith(SomePos), Some);
  EXPECT_EQ(Some.unionWith(SomeNeg), Some);
  EXPECT_EQ(Finite.unionWith(One), Finite);
  EXPECT_EQ(Some.unionWith(Zero), Some);
  EXPECT_EQ(Finite.unionWith(PosInf).unionWith(NegInf).unionWith(NaN), Full);
  EXPECT_EQ(PosZero.unionWith(NegZero), Zero);
  EXPECT_EQ(NaN.unionWith(SNaN), NaN);
  EXPECT_EQ(NaN.unionWith(QNaN), NaN);
  EXPECT_EQ(SNaN.unionWith(QNaN), NaN);
  EXPECT_EQ(
      ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(4.0))
          .unionWith(ConstantFPRange::getNonNaN(APFloat(3.0), APFloat(6.0))),
      ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(6.0)));
  EXPECT_EQ(
      ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(2.0))
          .unionWith(ConstantFPRange::getNonNaN(APFloat(5.0), APFloat(6.0))),
      ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(6.0)));
}

TEST_F(ConstantFPRangeTest, FPClassify) {
  EXPECT_EQ(Empty.classify(), fcNone);
  EXPECT_EQ(Full.classify(), fcAllFlags);
  EXPECT_EQ(Finite.classify(), fcFinite);
  EXPECT_EQ(Zero.classify(), fcZero);
  EXPECT_EQ(NaN.classify(), fcNan);
  EXPECT_EQ(SNaN.classify(), fcSNan);
  EXPECT_EQ(QNaN.classify(), fcQNan);
  EXPECT_EQ(One.classify(), fcPosNormal);
  EXPECT_EQ(Some.classify(), fcFinite);
  EXPECT_EQ(SomePos.classify(), fcPosFinite);
  EXPECT_EQ(SomeNeg.classify(), fcNegFinite);
  EXPECT_EQ(PosInf.classify(), fcPosInf);
  EXPECT_EQ(NegInf.classify(), fcNegInf);
  EXPECT_TRUE(SomePos.toKnownFPClass().cannotBeOrderedLessThanZero());
  EXPECT_EQ(Finite.getSignBit(), std::nullopt);
  EXPECT_EQ(PosZero.getSignBit(), false);
  EXPECT_EQ(NegZero.getSignBit(), true);
  EXPECT_EQ(SomePos.getSignBit(), false);
  EXPECT_EQ(SomeNeg.getSignBit(), true);
  EXPECT_EQ(SomePos.toKnownFPClass().SignBit, false);
  EXPECT_EQ(SomeNeg.toKnownFPClass().SignBit, true);

  EnumerateConstantFPRanges(
      [](const ConstantFPRange &CR) {
        unsigned Mask = fcNone;
        bool HasPos = false, HasNeg = false;
        EnumerateValuesInConstantFPRange(CR, [&](const APFloat &V) {
          Mask |= V.classify();
          if (V.isNegative())
            HasNeg = true;
          else
            HasPos = true;
        });

        std::optional<bool> SignBit = std::nullopt;
        if (HasPos != HasNeg)
          SignBit = HasNeg;

        EXPECT_EQ(SignBit, CR.getSignBit()) << CR;
        EXPECT_EQ(Mask, CR.classify()) << CR;
      },
      /*Exhaustive=*/true);
}

TEST_F(ConstantFPRangeTest, Print) {
  auto ToString = [](const ConstantFPRange &CR) {
    std::string Str;
    raw_string_ostream OS(Str);
    CR.print(OS);
    return Str;
  };

  EXPECT_EQ(ToString(Full), "full-set");
  EXPECT_EQ(ToString(Empty), "empty-set");
  EXPECT_EQ(ToString(NaN), "NaN");
  EXPECT_EQ(ToString(SNaN), "SNaN");
  EXPECT_EQ(ToString(QNaN), "QNaN");
  EXPECT_EQ(ToString(One), "[1, 1]");
  EXPECT_EQ(ToString(Some.unionWith(SNaN)), "[-3, 3] with SNaN");
}

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST_F(ConstantFPRangeTest, NonCanonicalEmptySet) {
  EXPECT_DEATH((void)(ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(0.0))),
               "Non-canonical form");
}
TEST_F(ConstantFPRangeTest, MismatchedSemantics) {
  EXPECT_DEATH((void)(ConstantFPRange::getNonNaN(APFloat(0.0), APFloat(1.0f))),
               "Should only use the same semantics");
  EXPECT_DEATH((void)(One.contains(APFloat(1.0f))),
               "Should only use the same semantics");
  ConstantFPRange OneF32 = ConstantFPRange(APFloat(1.0f));
  EXPECT_DEATH((void)(One.contains(OneF32)),
               "Should only use the same semantics");
  EXPECT_DEATH((void)(One.intersectWith(OneF32)),
               "Should only use the same semantics");
  EXPECT_DEATH((void)(One.unionWith(OneF32)),
               "Should only use the same semantics");
}
#endif
#endif

} // anonymous namespace
