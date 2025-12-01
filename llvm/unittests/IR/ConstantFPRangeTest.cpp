//===- ConstantRangeTest.cpp - ConstantRange tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ConstantFPRange.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class ConstantFPRangeTest : public ::testing::Test {
protected:
  static const fltSemantics &Sem;
  static ConstantFPRange Full;
  static ConstantFPRange Empty;
  static ConstantFPRange Finite;
  static ConstantFPRange NonNaN;
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
ConstantFPRange ConstantFPRangeTest::NonNaN =
    ConstantFPRange::getNonNaN(APFloat::IEEEdouble());
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

enum class SparseLevel {
  Dense,
  SpecialValuesWithAllPowerOfTwos,
  SpecialValuesOnly,
};

template <typename Fn>
static void EnumerateConstantFPRangesImpl(Fn TestFn, SparseLevel Level,
                                          bool MayBeQNaN, bool MayBeSNaN) {
  const fltSemantics &Sem = APFloat::Float8E4M3();
  APFloat PosInf = APFloat::getInf(Sem, /*Negative=*/false);
  APFloat NegInf = APFloat::getInf(Sem, /*Negative=*/true);
  TestFn(ConstantFPRange(PosInf, NegInf, MayBeQNaN, MayBeSNaN));

  if (Level != SparseLevel::Dense) {
    SmallVector<APFloat, 36> Values;
    Values.push_back(APFloat::getInf(Sem, /*Negative=*/true));
    Values.push_back(APFloat::getLargest(Sem, /*Negative=*/true));
    unsigned BitWidth = APFloat::semanticsSizeInBits(Sem);
    unsigned Exponents = APFloat::semanticsMaxExponent(Sem) -
                         APFloat::semanticsMinExponent(Sem) + 3;
    unsigned MantissaBits = APFloat::semanticsPrecision(Sem) - 1;
    if (Level == SparseLevel::SpecialValuesWithAllPowerOfTwos) {
      // Add -2^(max exponent), -2^(max exponent-1), ..., -2^(min exponent)
      for (unsigned M = Exponents - 2; M != 0; --M)
        Values.push_back(
            APFloat(Sem, APInt(BitWidth, (M + Exponents) << MantissaBits)));
    }
    Values.push_back(APFloat::getSmallestNormalized(Sem, /*Negative=*/true));
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
static void EnumerateConstantFPRanges(Fn TestFn, SparseLevel Level,
                                      bool IgnoreSNaNs = false) {
  EnumerateConstantFPRangesImpl(TestFn, Level, /*MayBeQNaN=*/false,
                                /*MayBeSNaN=*/false);
  EnumerateConstantFPRangesImpl(TestFn, Level, /*MayBeQNaN=*/true,
                                /*MayBeSNaN=*/false);
  if (IgnoreSNaNs)
    return;
  EnumerateConstantFPRangesImpl(TestFn, Level, /*MayBeQNaN=*/false,
                                /*MayBeSNaN=*/true);
  EnumerateConstantFPRangesImpl(TestFn, Level, /*MayBeQNaN=*/true,
                                /*MayBeSNaN=*/true);
}

template <typename Fn>
static void EnumerateTwoInterestingConstantFPRanges(Fn TestFn,
                                                    SparseLevel Level) {
  EnumerateConstantFPRanges(
      [&](const ConstantFPRange &CR1) {
        EnumerateConstantFPRanges(
            [&](const ConstantFPRange &CR2) { TestFn(CR1, CR2); }, Level,
            /*IgnoreSNaNs=*/true);
      },
      Level, /*IgnoreSNaNs=*/true);
}

template <typename Fn>
static void EnumerateValuesInConstantFPRange(const ConstantFPRange &CR,
                                             Fn TestFn, bool IgnoreNaNPayload) {
  const fltSemantics &Sem = CR.getSemantics();
  if (IgnoreNaNPayload) {
    if (CR.containsSNaN()) {
      TestFn(APFloat::getSNaN(Sem, false));
      TestFn(APFloat::getSNaN(Sem, true));
    }
    if (CR.containsQNaN()) {
      TestFn(APFloat::getQNaN(Sem, false));
      TestFn(APFloat::getQNaN(Sem, true));
    }
    if (CR.isNaNOnly())
      return;
    APFloat Lower = CR.getLower();
    const APFloat &Upper = CR.getUpper();
    auto Next = [&](APFloat &V) {
      if (V.bitwiseIsEqual(Upper))
        return false;
      strictNext(V);
      return true;
    };
    do
      TestFn(Lower);
    while (Next(Lower));
  } else {
    unsigned Bits = APFloat::semanticsSizeInBits(Sem);
    assert(Bits < 32 && "Too many bits");
    for (unsigned I = 0, E = (1U << Bits) - 1; I != E; ++I) {
      APFloat V(Sem, APInt(Bits, I));
      if (CR.contains(V))
        TestFn(V);
    }
  }
}

template <typename Fn>
static bool AnyOfValueInConstantFPRange(const ConstantFPRange &CR, Fn TestFn,
                                        bool IgnoreNaNPayload) {
  const fltSemantics &Sem = CR.getSemantics();
  if (IgnoreNaNPayload) {
    if (CR.containsSNaN()) {
      if (TestFn(APFloat::getSNaN(Sem, false)))
        return true;
      if (TestFn(APFloat::getSNaN(Sem, true)))
        return true;
    }
    if (CR.containsQNaN()) {
      if (TestFn(APFloat::getQNaN(Sem, false)))
        return true;
      if (TestFn(APFloat::getQNaN(Sem, true)))
        return true;
    }
    if (CR.isNaNOnly())
      return false;
    APFloat Lower = CR.getLower();
    const APFloat &Upper = CR.getUpper();
    auto Next = [&](APFloat &V) {
      if (V.bitwiseIsEqual(Upper))
        return false;
      strictNext(V);
      return true;
    };
    do {
      if (TestFn(Lower))
        return true;
    } while (Next(Lower));
  } else {
    unsigned Bits = APFloat::semanticsSizeInBits(Sem);
    assert(Bits < 32 && "Too many bits");
    for (unsigned I = 0, E = (1U << Bits) - 1; I != E; ++I) {
      APFloat V(Sem, APInt(Bits, I));
      if (CR.contains(V) && TestFn(V))
        return true;
    }
  }
  return false;
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
  ConstantFPRange PosZeroOrNaN = PosZero.unionWith(NaN);
  EXPECT_EQ(*PosZeroOrNaN.getSingleElement(/*ExcludesNaN=*/true),
            APFloat::getZero(Sem));

  EXPECT_FALSE(Full.isSingleElement());
  EXPECT_FALSE(Empty.isSingleElement());
  EXPECT_TRUE(One.isSingleElement());
  EXPECT_FALSE(Some.isSingleElement());
  EXPECT_FALSE(Zero.isSingleElement());
  EXPECT_TRUE(PosZeroOrNaN.isSingleElement(/*ExcludesNaN=*/true));
}

TEST_F(ConstantFPRangeTest, ExhaustivelyEnumerate) {
  constexpr unsigned NNaNValues = (1 << 8) - 2 * ((1 << 3) - 1);
  constexpr unsigned Expected = 4 * ((NNaNValues + 1) * NNaNValues / 2 + 1);
  unsigned Count = 0;
  EnumerateConstantFPRanges([&](const ConstantFPRange &) { ++Count; },
                            SparseLevel::Dense);
  EXPECT_EQ(Expected, Count);
}

TEST_F(ConstantFPRangeTest, Enumerate) {
  constexpr unsigned NNaNValues = 2 * ((1 << 4) - 2 + 5);
  constexpr unsigned Expected = 4 * ((NNaNValues + 1) * NNaNValues / 2 + 1);
  unsigned Count = 0;
  EnumerateConstantFPRanges([&](const ConstantFPRange &) { ++Count; },
                            SparseLevel::SpecialValuesWithAllPowerOfTwos);
  EXPECT_EQ(Expected, Count);
}

TEST_F(ConstantFPRangeTest, EnumerateWithSpecialValuesOnly) {
  constexpr unsigned NNaNValues = 2 * 5;
  constexpr unsigned Expected = 4 * ((NNaNValues + 1) * NNaNValues / 2 + 1);
  unsigned Count = 0;
  EnumerateConstantFPRanges([&](const ConstantFPRange &) { ++Count; },
                            SparseLevel::SpecialValuesOnly);
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
  EXPECT_EQ(Finite.getSignBit(), std::nullopt);
  EXPECT_EQ(PosZero.getSignBit(), false);
  EXPECT_EQ(NegZero.getSignBit(), true);
  EXPECT_EQ(SomePos.getSignBit(), false);
  EXPECT_EQ(SomeNeg.getSignBit(), true);

#if defined(EXPENSIVE_CHECKS)
  EnumerateConstantFPRanges(
      [](const ConstantFPRange &CR) {
        unsigned Mask = fcNone;
        bool HasPos = false, HasNeg = false;
        EnumerateValuesInConstantFPRange(
            CR,
            [&](const APFloat &V) {
              Mask |= V.classify();
              if (V.isNegative())
                HasNeg = true;
              else
                HasPos = true;
            },
            /*IgnoreNaNPayload=*/true);

        std::optional<bool> SignBit = std::nullopt;
        if (HasPos != HasNeg)
          SignBit = HasNeg;

        EXPECT_EQ(SignBit, CR.getSignBit()) << CR;
        EXPECT_EQ(Mask, CR.classify()) << CR;
      },
      SparseLevel::Dense);
#endif
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

TEST_F(ConstantFPRangeTest, makeAllowedFCmpRegion) {
  EXPECT_EQ(ConstantFPRange::makeAllowedFCmpRegion(
                FCmpInst::FCMP_OLE,
                ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(2.0))),
            ConstantFPRange::getNonNaN(APFloat::getInf(Sem, /*Negative=*/true),
                                       APFloat(2.0)));
  EXPECT_EQ(
      ConstantFPRange::makeAllowedFCmpRegion(
          FCmpInst::FCMP_OLT,
          ConstantFPRange::getNonNaN(APFloat(1.0),
                                     APFloat::getInf(Sem, /*Negative=*/false))),
      ConstantFPRange::getNonNaN(APFloat::getInf(Sem, /*Negative=*/true),
                                 APFloat::getLargest(Sem, /*Negative=*/false)));
  EXPECT_EQ(
      ConstantFPRange::makeAllowedFCmpRegion(
          FCmpInst::FCMP_OGT,
          ConstantFPRange::getNonNaN(APFloat::getZero(Sem, /*Negative=*/true),
                                     APFloat(2.0))),
      ConstantFPRange::getNonNaN(APFloat::getSmallest(Sem, /*Negative=*/false),
                                 APFloat::getInf(Sem, /*Negative=*/false)));
  EXPECT_EQ(ConstantFPRange::makeAllowedFCmpRegion(
                FCmpInst::FCMP_OGE,
                ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(2.0))),
            ConstantFPRange::getNonNaN(
                APFloat(1.0), APFloat::getInf(Sem, /*Negative=*/false)));
  EXPECT_EQ(ConstantFPRange::makeAllowedFCmpRegion(
                FCmpInst::FCMP_OEQ,
                ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(2.0))),
            ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(2.0)));

#if defined(EXPENSIVE_CHECKS)
  for (auto Pred : FCmpInst::predicates()) {
    EnumerateConstantFPRanges(
        [Pred](const ConstantFPRange &CR) {
          ConstantFPRange Res =
              ConstantFPRange::makeAllowedFCmpRegion(Pred, CR);
          ConstantFPRange Optimal =
              ConstantFPRange::getEmpty(CR.getSemantics());
          EnumerateValuesInConstantFPRange(
              ConstantFPRange::getFull(CR.getSemantics()),
              [&](const APFloat &V) {
                if (AnyOfValueInConstantFPRange(
                        CR,
                        [&](const APFloat &U) {
                          return FCmpInst::compare(V, U, Pred);
                        },
                        /*IgnoreNaNPayload=*/true))
                  Optimal = Optimal.unionWith(ConstantFPRange(V));
              },
              /*IgnoreNaNPayload=*/true);

          EXPECT_TRUE(Res.contains(Optimal))
              << "Wrong result for makeAllowedFCmpRegion(" << Pred << ", " << CR
              << "). Expected " << Optimal << ", but got " << Res;
          EXPECT_EQ(Res, Optimal)
              << "Suboptimal result for makeAllowedFCmpRegion(" << Pred << ", "
              << CR << ")";
        },
        SparseLevel::SpecialValuesWithAllPowerOfTwos);
  }
#endif
}

TEST_F(ConstantFPRangeTest, makeSatisfyingFCmpRegion) {
  EXPECT_EQ(ConstantFPRange::makeSatisfyingFCmpRegion(
                FCmpInst::FCMP_OLE,
                ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(2.0))),
            ConstantFPRange::getNonNaN(APFloat::getInf(Sem, /*Negative=*/true),
                                       APFloat(1.0)));
  EXPECT_EQ(
      ConstantFPRange::makeSatisfyingFCmpRegion(
          FCmpInst::FCMP_OLT, ConstantFPRange::getNonNaN(
                                  APFloat::getSmallest(Sem, /*Negative=*/false),
                                  APFloat::getInf(Sem, /*Negative=*/false))),
      ConstantFPRange::getNonNaN(APFloat::getInf(Sem, /*Negative=*/true),
                                 APFloat::getZero(Sem, /*Negative=*/false)));
  EXPECT_EQ(
      ConstantFPRange::makeSatisfyingFCmpRegion(
          FCmpInst::FCMP_OGT, ConstantFPRange::getNonNaN(
                                  APFloat::getZero(Sem, /*Negative=*/true),
                                  APFloat::getZero(Sem, /*Negative=*/false))),
      ConstantFPRange::getNonNaN(APFloat::getSmallest(Sem, /*Negative=*/false),
                                 APFloat::getInf(Sem, /*Negative=*/false)));
  EXPECT_EQ(ConstantFPRange::makeSatisfyingFCmpRegion(
                FCmpInst::FCMP_OGE,
                ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(2.0))),
            ConstantFPRange::getNonNaN(
                APFloat(2.0), APFloat::getInf(Sem, /*Negative=*/false)));
  EXPECT_EQ(ConstantFPRange::makeSatisfyingFCmpRegion(
                FCmpInst::FCMP_OEQ,
                ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(2.0))),
            ConstantFPRange::getEmpty(Sem));
  EXPECT_EQ(ConstantFPRange::makeSatisfyingFCmpRegion(
                FCmpInst::FCMP_OEQ,
                ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(1.0))),
            ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(1.0)));

#if defined(EXPENSIVE_CHECKS)
  for (auto Pred : FCmpInst::predicates()) {
    EnumerateConstantFPRanges(
        [Pred](const ConstantFPRange &CR) {
          ConstantFPRange Res =
              ConstantFPRange::makeSatisfyingFCmpRegion(Pred, CR);
          // Super set of the optimal set excluding NaNs
          ConstantFPRange SuperSet(CR.getSemantics());
          bool ContainsSNaN = false;
          bool ContainsQNaN = false;
          unsigned NonNaNValsInOptimalSet = 0;
          EnumerateValuesInConstantFPRange(
              ConstantFPRange::getFull(CR.getSemantics()),
              [&](const APFloat &V) {
                if (AnyOfValueInConstantFPRange(
                        CR,
                        [&](const APFloat &U) {
                          return !FCmpInst::compare(V, U, Pred);
                        },
                        /*IgnoreNaNPayload=*/true)) {
                  EXPECT_FALSE(Res.contains(V))
                      << "Wrong result for makeSatisfyingFCmpRegion(" << Pred
                      << ", " << CR << "). The result " << Res
                      << " should not contain " << V;
                } else {
                  if (V.isNaN()) {
                    if (V.isSignaling())
                      ContainsSNaN = true;
                    else
                      ContainsQNaN = true;
                  } else {
                    SuperSet = SuperSet.unionWith(ConstantFPRange(V));
                    ++NonNaNValsInOptimalSet;
                  }
                }
              },
              /*IgnoreNaNPayload=*/true);

          // Check optimality

          // The usefullness of making the result optimal for one/une is
          // questionable.
          if (Pred == FCmpInst::FCMP_ONE || Pred == FCmpInst::FCMP_UNE)
            return;

          EXPECT_FALSE(ContainsSNaN && !Res.containsSNaN())
              << "Suboptimal result for makeSatisfyingFCmpRegion(" << Pred
              << ", " << CR << "), should contain SNaN, but got " << Res;
          EXPECT_FALSE(ContainsQNaN && !Res.containsQNaN())
              << "Suboptimal result for makeSatisfyingFCmpRegion(" << Pred
              << ", " << CR << "), should contain QNaN, but got " << Res;

          // We only care about the cases where the result is representable by
          // ConstantFPRange.
          unsigned NonNaNValsInSuperSet = 0;
          EnumerateValuesInConstantFPRange(
              SuperSet,
              [&](const APFloat &V) {
                if (!V.isNaN())
                  ++NonNaNValsInSuperSet;
              },
              /*IgnoreNaNPayload=*/true);

          if (NonNaNValsInSuperSet == NonNaNValsInOptimalSet) {
            ConstantFPRange Optimal =
                ConstantFPRange(SuperSet.getLower(), SuperSet.getUpper(),
                                ContainsQNaN, ContainsSNaN);
            EXPECT_EQ(Res, Optimal)
                << "Suboptimal result for makeSatisfyingFCmpRegion(" << Pred
                << ", " << CR << ")";
          }
        },
        SparseLevel::SpecialValuesWithAllPowerOfTwos);
  }
#endif
}

TEST_F(ConstantFPRangeTest, fcmp) {
  std::vector<ConstantFPRange> InterestingRanges;
  const fltSemantics &Sem = APFloat::Float8E4M3();
  auto FpImm = [&](double V) {
    bool ignored;
    APFloat APF(V);
    APF.convert(Sem, APFloat::rmNearestTiesToEven, &ignored);
    return APF;
  };

  InterestingRanges.push_back(ConstantFPRange::getEmpty(Sem));
  InterestingRanges.push_back(ConstantFPRange::getFull(Sem));
  InterestingRanges.push_back(ConstantFPRange::getFinite(Sem));
  InterestingRanges.push_back(ConstantFPRange(FpImm(1.0)));
  InterestingRanges.push_back(
      ConstantFPRange(APFloat::getZero(Sem, /*Negative=*/false)));
  InterestingRanges.push_back(
      ConstantFPRange(APFloat::getZero(Sem, /*Negative=*/true)));
  InterestingRanges.push_back(
      ConstantFPRange(APFloat::getInf(Sem, /*Negative=*/false)));
  InterestingRanges.push_back(
      ConstantFPRange(APFloat::getInf(Sem, /*Negative=*/true)));
  InterestingRanges.push_back(
      ConstantFPRange(APFloat::getSmallest(Sem, /*Negative=*/false)));
  InterestingRanges.push_back(
      ConstantFPRange(APFloat::getSmallest(Sem, /*Negative=*/true)));
  InterestingRanges.push_back(
      ConstantFPRange(APFloat::getLargest(Sem, /*Negative=*/false)));
  InterestingRanges.push_back(
      ConstantFPRange(APFloat::getLargest(Sem, /*Negative=*/true)));
  InterestingRanges.push_back(
      ConstantFPRange::getNaNOnly(Sem, /*MayBeQNaN=*/true, /*MayBeSNaN=*/true));
  InterestingRanges.push_back(
      ConstantFPRange::getNonNaN(FpImm(0.0), FpImm(1.0)));
  InterestingRanges.push_back(
      ConstantFPRange::getNonNaN(FpImm(2.0), FpImm(3.0)));
  InterestingRanges.push_back(
      ConstantFPRange::getNonNaN(FpImm(-1.0), FpImm(1.0)));
  InterestingRanges.push_back(
      ConstantFPRange::getNonNaN(FpImm(-1.0), FpImm(-0.0)));
  InterestingRanges.push_back(ConstantFPRange::getNonNaN(
      APFloat::getInf(Sem, /*Negative=*/true), FpImm(-1.0)));
  InterestingRanges.push_back(ConstantFPRange::getNonNaN(
      FpImm(1.0), APFloat::getInf(Sem, /*Negative=*/false)));

  for (auto &LHS : InterestingRanges) {
    for (auto &RHS : InterestingRanges) {
      for (auto Pred : FCmpInst::predicates()) {
        if (LHS.fcmp(Pred, RHS)) {
          EnumerateValuesInConstantFPRange(
              LHS,
              [&](const APFloat &LHSC) {
                EnumerateValuesInConstantFPRange(
                    RHS,
                    [&](const APFloat &RHSC) {
                      EXPECT_TRUE(FCmpInst::compare(LHSC, RHSC, Pred))
                          << LHS << " " << Pred << " " << RHS
                          << " doesn't hold";
                    },
                    /*IgnoreNaNPayload=*/true);
              },
              /*IgnoreNaNPayload=*/true);
        }
      }
    }
  }
}

TEST_F(ConstantFPRangeTest, makeExactFCmpRegion) {
  for (auto Pred : FCmpInst::predicates()) {
    EnumerateValuesInConstantFPRange(
        ConstantFPRange::getFull(APFloat::Float8E4M3()),
        [Pred](const APFloat &V) {
          std::optional<ConstantFPRange> Res =
              ConstantFPRange::makeExactFCmpRegion(Pred, V);
          ConstantFPRange Allowed =
              ConstantFPRange::makeAllowedFCmpRegion(Pred, ConstantFPRange(V));
          ConstantFPRange Satisfying =
              ConstantFPRange::makeSatisfyingFCmpRegion(Pred,
                                                        ConstantFPRange(V));
          if (Allowed == Satisfying)
            EXPECT_EQ(Res, Allowed) << "Wrong result for makeExactFCmpRegion("
                                    << Pred << ", " << V << ").";
          else
            EXPECT_FALSE(Res.has_value())
                << "Wrong result for makeExactFCmpRegion(" << Pred << ", " << V
                << ").";
        },
        /*IgnoreNaNPayload=*/true);
  }
}

TEST_F(ConstantFPRangeTest, abs) {
  EXPECT_EQ(Full.abs(),
            ConstantFPRange(APFloat::getZero(Sem, /*Negative=*/false),
                            APFloat::getInf(Sem, /*Negative=*/false),
                            /*MayBeQNaN=*/true,
                            /*MayBeSNaN=*/true));
  EXPECT_EQ(Empty.abs(), Empty);
  EXPECT_EQ(Zero.abs(), PosZero);
  EXPECT_EQ(PosInf.abs(), PosInf);
  EXPECT_EQ(NegInf.abs(), PosInf);
  EXPECT_EQ(Some.abs(), SomePos);
  EXPECT_EQ(SomeNeg.abs(), SomePos);
  EXPECT_EQ(NaN.abs(), NaN);
  EXPECT_EQ(ConstantFPRange::getNonNaN(APFloat(-2.0), APFloat(3.0)).abs(),
            ConstantFPRange::getNonNaN(APFloat(0.0), APFloat(3.0)));
  EXPECT_EQ(ConstantFPRange::getNonNaN(APFloat(-3.0), APFloat(2.0)).abs(),
            ConstantFPRange::getNonNaN(APFloat(0.0), APFloat(3.0)));
}

TEST_F(ConstantFPRangeTest, negate) {
  EXPECT_EQ(Full.negate(), Full);
  EXPECT_EQ(Empty.negate(), Empty);
  EXPECT_EQ(Zero.negate(), Zero);
  EXPECT_EQ(PosInf.negate(), NegInf);
  EXPECT_EQ(NegInf.negate(), PosInf);
  EXPECT_EQ(Some.negate(), Some);
  EXPECT_EQ(SomePos.negate(), SomeNeg);
  EXPECT_EQ(SomeNeg.negate(), SomePos);
  EXPECT_EQ(NaN.negate(), NaN);
  EXPECT_EQ(ConstantFPRange::getNonNaN(APFloat(-2.0), APFloat(3.0)).negate(),
            ConstantFPRange::getNonNaN(APFloat(-3.0), APFloat(2.0)));
  EXPECT_EQ(ConstantFPRange::getNonNaN(APFloat(-3.0), APFloat(2.0)).negate(),
            ConstantFPRange::getNonNaN(APFloat(-2.0), APFloat(3.0)));
}

TEST_F(ConstantFPRangeTest, getWithout) {
  EXPECT_EQ(Full.getWithoutNaN(), NonNaN);
  EXPECT_EQ(NaN.getWithoutNaN(), Empty);

  EXPECT_EQ(NaN.getWithoutInf(), NaN);
  EXPECT_EQ(PosInf.getWithoutInf(), Empty);
  EXPECT_EQ(NegInf.getWithoutInf(), Empty);
  EXPECT_EQ(NonNaN.getWithoutInf(), Finite);
  EXPECT_EQ(Zero.getWithoutInf(), Zero);
  EXPECT_EQ(ConstantFPRange::getNonNaN(APFloat::getInf(Sem, /*Negative=*/true),
                                       APFloat(3.0))
                .getWithoutInf(),
            ConstantFPRange::getNonNaN(
                APFloat::getLargest(Sem, /*Negative=*/true), APFloat(3.0)));
}

TEST_F(ConstantFPRangeTest, cast) {
  const fltSemantics &F16Sem = APFloat::IEEEhalf();
  const fltSemantics &BF16Sem = APFloat::BFloat();
  const fltSemantics &F32Sem = APFloat::IEEEsingle();
  const fltSemantics &F8NanOnlySem = APFloat::Float8E4M3FN();
  // normal -> normal (exact)
  EXPECT_EQ(ConstantFPRange::getNonNaN(APFloat(1.0), APFloat(2.0)).cast(F32Sem),
            ConstantFPRange::getNonNaN(APFloat(1.0f), APFloat(2.0f)));
  EXPECT_EQ(
      ConstantFPRange::getNonNaN(APFloat(-2.0f), APFloat(-1.0f)).cast(Sem),
      ConstantFPRange::getNonNaN(APFloat(-2.0), APFloat(-1.0)));
  // normal -> normal (inexact)
  EXPECT_EQ(
      ConstantFPRange::getNonNaN(APFloat(3.141592653589793),
                                 APFloat(6.283185307179586))
          .cast(F32Sem),
      ConstantFPRange::getNonNaN(APFloat(3.14159274f), APFloat(6.28318548f)));
  // normal -> subnormal
  EXPECT_EQ(ConstantFPRange::getNonNaN(APFloat(-5e-8), APFloat(5e-8))
                .cast(F16Sem)
                .classify(),
            fcSubnormal | fcZero);
  // normal -> zero
  EXPECT_EQ(ConstantFPRange::getNonNaN(
                APFloat::getSmallestNormalized(Sem, /*Negative=*/true),
                APFloat::getSmallestNormalized(Sem, /*Negative=*/false))
                .cast(F32Sem)
                .classify(),
            fcZero);
  // normal -> inf
  EXPECT_EQ(ConstantFPRange::getNonNaN(APFloat(-65536.0), APFloat(65536.0))
                .cast(F16Sem),
            ConstantFPRange::getNonNaN(F16Sem));
  // nan -> qnan
  EXPECT_EQ(
      ConstantFPRange::getNaNOnly(Sem, /*MayBeQNaN=*/true, /*MayBeSNaN=*/false)
          .cast(F32Sem),
      ConstantFPRange::getNaNOnly(F32Sem, /*MayBeQNaN=*/true,
                                  /*MayBeSNaN=*/false));
  EXPECT_EQ(
      ConstantFPRange::getNaNOnly(Sem, /*MayBeQNaN=*/false, /*MayBeSNaN=*/true)
          .cast(F32Sem),
      ConstantFPRange::getNaNOnly(F32Sem, /*MayBeQNaN=*/true,
                                  /*MayBeSNaN=*/false));
  EXPECT_EQ(
      ConstantFPRange::getNaNOnly(Sem, /*MayBeQNaN=*/true, /*MayBeSNaN=*/true)
          .cast(F32Sem),
      ConstantFPRange::getNaNOnly(F32Sem, /*MayBeQNaN=*/true,
                                  /*MayBeSNaN=*/false));
  // For BF16 -> F32, signaling bit is still lost.
  EXPECT_EQ(ConstantFPRange::getNaNOnly(BF16Sem, /*MayBeQNaN=*/true,
                                        /*MayBeSNaN=*/true)
                .cast(F32Sem),
            ConstantFPRange::getNaNOnly(F32Sem, /*MayBeQNaN=*/true,
                                        /*MayBeSNaN=*/false));
  // inf -> nan only (return full set for now)
  EXPECT_EQ(ConstantFPRange::getNonNaN(APFloat::getInf(Sem, /*Negative=*/true),
                                       APFloat::getInf(Sem, /*Negative=*/false))
                .cast(F8NanOnlySem),
            ConstantFPRange::getFull(F8NanOnlySem));
  // other rounding modes
  EXPECT_EQ(
      ConstantFPRange::getNonNaN(APFloat::getSmallest(Sem, /*Negative=*/true),
                                 APFloat::getSmallest(Sem, /*Negative=*/false))
          .cast(F32Sem, APFloat::rmTowardNegative),
      ConstantFPRange::getNonNaN(
          APFloat::getSmallest(F32Sem, /*Negative=*/true),
          APFloat::getZero(F32Sem, /*Negative=*/false)));
  EXPECT_EQ(
      ConstantFPRange::getNonNaN(APFloat::getSmallest(Sem, /*Negative=*/true),
                                 APFloat::getSmallest(Sem, /*Negative=*/false))
          .cast(F32Sem, APFloat::rmTowardPositive),
      ConstantFPRange::getNonNaN(
          APFloat::getZero(F32Sem, /*Negative=*/true),
          APFloat::getSmallest(F32Sem, /*Negative=*/false)));
  EXPECT_EQ(
      ConstantFPRange::getNonNaN(
          APFloat::getSmallestNormalized(Sem, /*Negative=*/true),
          APFloat::getSmallestNormalized(Sem, /*Negative=*/false))
          .cast(F32Sem, APFloat::rmTowardZero),
      ConstantFPRange::getNonNaN(APFloat::getZero(F32Sem, /*Negative=*/true),
                                 APFloat::getZero(F32Sem, /*Negative=*/false)));

  EnumerateValuesInConstantFPRange(
      ConstantFPRange::getFull(APFloat::Float8E4M3()),
      [&](const APFloat &V) {
        bool LosesInfo = false;

        APFloat DoubleV = V;
        DoubleV.convert(Sem, APFloat::rmNearestTiesToEven, &LosesInfo);
        ConstantFPRange DoubleCR = ConstantFPRange(V).cast(Sem);
        EXPECT_TRUE(DoubleCR.contains(DoubleV))
            << "Casting " << V << " to double failed. " << DoubleCR
            << " doesn't contain " << DoubleV;

        auto &FP4Sem = APFloat::Float4E2M1FN();
        APFloat FP4V = V;
        FP4V.convert(FP4Sem, APFloat::rmNearestTiesToEven, &LosesInfo);
        ConstantFPRange FP4CR = ConstantFPRange(V).cast(FP4Sem);
        EXPECT_TRUE(FP4CR.contains(FP4V))
            << "Casting " << V << " to FP4E2M1FN failed. " << FP4CR
            << " doesn't contain " << FP4V;
      },
      /*IgnoreNaNPayload=*/true);
}

TEST_F(ConstantFPRangeTest, add) {
  EXPECT_EQ(Full.add(Full), NonNaN.unionWith(QNaN));
  EXPECT_EQ(Full.add(Empty), Empty);
  EXPECT_EQ(Empty.add(Full), Empty);
  EXPECT_EQ(Empty.add(Empty), Empty);
  EXPECT_EQ(One.add(One), ConstantFPRange(APFloat(2.0)));
  EXPECT_EQ(Some.add(Some),
            ConstantFPRange::getNonNaN(APFloat(-6.0), APFloat(6.0)));
  EXPECT_EQ(SomePos.add(SomeNeg),
            ConstantFPRange::getNonNaN(APFloat(-3.0), APFloat(3.0)));
  EXPECT_EQ(PosInf.add(PosInf), PosInf);
  EXPECT_EQ(NegInf.add(NegInf), NegInf);
  EXPECT_EQ(PosInf.add(Finite.unionWith(PosInf)), PosInf);
  EXPECT_EQ(NegInf.add(Finite.unionWith(NegInf)), NegInf);
  EXPECT_EQ(PosInf.add(Finite.unionWith(NegInf)), PosInf.unionWith(QNaN));
  EXPECT_EQ(NegInf.add(Finite.unionWith(PosInf)), NegInf.unionWith(QNaN));
  EXPECT_EQ(PosInf.add(NegInf), QNaN);
  EXPECT_EQ(NegInf.add(PosInf), QNaN);
  EXPECT_EQ(PosZero.add(NegZero), PosZero);
  EXPECT_EQ(PosZero.add(Zero), PosZero);
  EXPECT_EQ(NegZero.add(NegZero), NegZero);
  EXPECT_EQ(NegZero.add(Zero), Zero);
  EXPECT_EQ(NaN.add(NaN), QNaN);
  EXPECT_EQ(NaN.add(Finite), QNaN);
  EXPECT_EQ(NonNaN.unionWith(NaN).add(NonNaN), NonNaN.unionWith(QNaN));
  EXPECT_EQ(PosInf.unionWith(QNaN).add(PosInf), PosInf.unionWith(QNaN));
  EXPECT_EQ(PosInf.unionWith(NaN).add(ConstantFPRange(APFloat(24.0))),
            PosInf.unionWith(QNaN));

#if defined(EXPENSIVE_CHECKS)
  EnumerateTwoInterestingConstantFPRanges(
      [](const ConstantFPRange &LHS, const ConstantFPRange &RHS) {
        ConstantFPRange Res = LHS.add(RHS);
        ConstantFPRange Expected =
            ConstantFPRange::getEmpty(LHS.getSemantics());
        EnumerateValuesInConstantFPRange(
            LHS,
            [&](const APFloat &LHSC) {
              EnumerateValuesInConstantFPRange(
                  RHS,
                  [&](const APFloat &RHSC) {
                    APFloat Sum = LHSC + RHSC;
                    EXPECT_TRUE(Res.contains(Sum))
                        << "Wrong result for " << LHS << " + " << RHS
                        << ". The result " << Res << " should contain " << Sum;
                    if (!Expected.contains(Sum))
                      Expected = Expected.unionWith(ConstantFPRange(Sum));
                  },
                  /*IgnoreNaNPayload=*/true);
            },
            /*IgnoreNaNPayload=*/true);
        EXPECT_EQ(Res, Expected)
            << "Suboptimal result for " << LHS << " + " << RHS << ". Expected "
            << Expected << ", but got " << Res;
      },
      SparseLevel::SpecialValuesOnly);
#endif
}

TEST_F(ConstantFPRangeTest, sub) {
  EXPECT_EQ(Full.sub(Full), NonNaN.unionWith(QNaN));
  EXPECT_EQ(Full.sub(Empty), Empty);
  EXPECT_EQ(Empty.sub(Full), Empty);
  EXPECT_EQ(Empty.sub(Empty), Empty);
  EXPECT_EQ(One.sub(One), ConstantFPRange(APFloat(0.0)));
  EXPECT_EQ(Some.sub(Some),
            ConstantFPRange::getNonNaN(APFloat(-6.0), APFloat(6.0)));
  EXPECT_EQ(SomePos.sub(SomeNeg),
            ConstantFPRange::getNonNaN(APFloat(0.0), APFloat(6.0)));
  EXPECT_EQ(PosInf.sub(NegInf), PosInf);
  EXPECT_EQ(NegInf.sub(PosInf), NegInf);
  EXPECT_EQ(PosInf.sub(Finite.unionWith(NegInf)), PosInf);
  EXPECT_EQ(NegInf.sub(Finite.unionWith(PosInf)), NegInf);
  EXPECT_EQ(PosInf.sub(Finite.unionWith(PosInf)), PosInf.unionWith(QNaN));
  EXPECT_EQ(NegInf.sub(Finite.unionWith(NegInf)), NegInf.unionWith(QNaN));
  EXPECT_EQ(PosInf.sub(PosInf), QNaN);
  EXPECT_EQ(NegInf.sub(NegInf), QNaN);
  EXPECT_EQ(PosZero.sub(NegZero), PosZero);
  EXPECT_EQ(PosZero.sub(Zero), PosZero);
  EXPECT_EQ(NegZero.sub(NegZero), PosZero);
  EXPECT_EQ(NegZero.sub(PosZero), NegZero);
  EXPECT_EQ(NegZero.sub(Zero), Zero);
  EXPECT_EQ(NaN.sub(NaN), QNaN);
  EXPECT_EQ(NaN.add(Finite), QNaN);

#if defined(EXPENSIVE_CHECKS)
  EnumerateTwoInterestingConstantFPRanges(
      [](const ConstantFPRange &LHS, const ConstantFPRange &RHS) {
        ConstantFPRange Res = LHS.sub(RHS);
        ConstantFPRange Expected =
            ConstantFPRange::getEmpty(LHS.getSemantics());
        EnumerateValuesInConstantFPRange(
            LHS,
            [&](const APFloat &LHSC) {
              EnumerateValuesInConstantFPRange(
                  RHS,
                  [&](const APFloat &RHSC) {
                    APFloat Diff = LHSC - RHSC;
                    EXPECT_TRUE(Res.contains(Diff))
                        << "Wrong result for " << LHS << " - " << RHS
                        << ". The result " << Res << " should contain " << Diff;
                    if (!Expected.contains(Diff))
                      Expected = Expected.unionWith(ConstantFPRange(Diff));
                  },
                  /*IgnoreNaNPayload=*/true);
            },
            /*IgnoreNaNPayload=*/true);
        EXPECT_EQ(Res, Expected)
            << "Suboptimal result for " << LHS << " - " << RHS << ". Expected "
            << Expected << ", but got " << Res;
      },
      SparseLevel::SpecialValuesOnly);
#endif
}

TEST_F(ConstantFPRangeTest, mul) {
  EXPECT_EQ(Full.mul(Full), NonNaN.unionWith(QNaN));
  EXPECT_EQ(Full.mul(Empty), Empty);
  EXPECT_EQ(Empty.mul(Full), Empty);
  EXPECT_EQ(Empty.mul(Empty), Empty);
  EXPECT_EQ(One.mul(One), ConstantFPRange(APFloat(1.0)));
  EXPECT_EQ(Some.mul(Some),
            ConstantFPRange::getNonNaN(APFloat(-9.0), APFloat(9.0)));
  EXPECT_EQ(SomePos.mul(SomeNeg),
            ConstantFPRange::getNonNaN(APFloat(-9.0), APFloat(-0.0)));
  EXPECT_EQ(PosInf.mul(PosInf), PosInf);
  EXPECT_EQ(NegInf.mul(NegInf), PosInf);
  EXPECT_EQ(PosInf.mul(Finite), NonNaN.unionWith(QNaN));
  EXPECT_EQ(NegInf.mul(Finite), NonNaN.unionWith(QNaN));
  EXPECT_EQ(PosInf.mul(NegInf), NegInf);
  EXPECT_EQ(NegInf.mul(PosInf), NegInf);
  EXPECT_EQ(PosZero.mul(NegZero), NegZero);
  EXPECT_EQ(PosZero.mul(Zero), Zero);
  EXPECT_EQ(NegZero.mul(NegZero), PosZero);
  EXPECT_EQ(NegZero.mul(Zero), Zero);
  EXPECT_EQ(NaN.mul(NaN), QNaN);
  EXPECT_EQ(NaN.mul(Finite), QNaN);

#if defined(EXPENSIVE_CHECKS)
  EnumerateTwoInterestingConstantFPRanges(
      [](const ConstantFPRange &LHS, const ConstantFPRange &RHS) {
        ConstantFPRange Res = LHS.mul(RHS);
        ConstantFPRange Expected =
            ConstantFPRange::getEmpty(LHS.getSemantics());
        EnumerateValuesInConstantFPRange(
            LHS,
            [&](const APFloat &LHSC) {
              EnumerateValuesInConstantFPRange(
                  RHS,
                  [&](const APFloat &RHSC) {
                    APFloat Prod = LHSC * RHSC;
                    EXPECT_TRUE(Res.contains(Prod))
                        << "Wrong result for " << LHS << " * " << RHS
                        << ". The result " << Res << " should contain " << Prod;
                    if (!Expected.contains(Prod))
                      Expected = Expected.unionWith(ConstantFPRange(Prod));
                  },
                  /*IgnoreNaNPayload=*/true);
            },
            /*IgnoreNaNPayload=*/true);
        EXPECT_EQ(Res, Expected)
            << "Suboptimal result for " << LHS << " * " << RHS << ". Expected "
            << Expected << ", but got " << Res;
      },
      SparseLevel::SpecialValuesOnly);
#endif
}

TEST_F(ConstantFPRangeTest, div) {
  EXPECT_EQ(Full.div(Full), NonNaN.unionWith(QNaN));
  EXPECT_EQ(Full.div(Empty), Empty);
  EXPECT_EQ(Empty.div(Full), Empty);
  EXPECT_EQ(Empty.div(Empty), Empty);
  EXPECT_EQ(One.div(One), ConstantFPRange(APFloat(1.0)));
  EXPECT_EQ(Some.div(Some), NonNaN.unionWith(QNaN));
  EXPECT_EQ(SomePos.div(SomeNeg),
            ConstantFPRange(APFloat::getInf(Sem, /*Negative=*/true),
                            APFloat::getZero(Sem, /*Negative=*/true),
                            /*MayBeQNaN=*/true, /*MayBeSNaN=*/false));
  EXPECT_EQ(PosInf.div(PosInf), QNaN);
  EXPECT_EQ(NegInf.div(NegInf), QNaN);
  EXPECT_EQ(PosInf.div(Finite), NonNaN);
  EXPECT_EQ(NegInf.div(Finite), NonNaN);
  EXPECT_EQ(PosInf.div(NegInf), QNaN);
  EXPECT_EQ(NegInf.div(PosInf), QNaN);
  EXPECT_EQ(Zero.div(Zero), QNaN);
  EXPECT_EQ(SomePos.div(PosInf), PosZero);
  EXPECT_EQ(SomeNeg.div(PosInf), NegZero);
  EXPECT_EQ(PosInf.div(SomePos), PosInf);
  EXPECT_EQ(NegInf.div(SomeNeg), PosInf);
  EXPECT_EQ(NegInf.div(Some), NonNaN);
  EXPECT_EQ(NaN.div(NaN), QNaN);
  EXPECT_EQ(NaN.div(Finite), QNaN);

#if defined(EXPENSIVE_CHECKS)
  EnumerateTwoInterestingConstantFPRanges(
      [](const ConstantFPRange &LHS, const ConstantFPRange &RHS) {
        ConstantFPRange Res = LHS.div(RHS);
        ConstantFPRange Expected =
            ConstantFPRange::getEmpty(LHS.getSemantics());
        EnumerateValuesInConstantFPRange(
            LHS,
            [&](const APFloat &LHSC) {
              EnumerateValuesInConstantFPRange(
                  RHS,
                  [&](const APFloat &RHSC) {
                    APFloat Val = LHSC / RHSC;
                    EXPECT_TRUE(Res.contains(Val))
                        << "Wrong result for " << LHS << " / " << RHS
                        << ". The result " << Res << " should contain " << Val;
                    if (!Expected.contains(Val))
                      Expected = Expected.unionWith(ConstantFPRange(Val));
                  },
                  /*IgnoreNaNPayload=*/true);
            },
            /*IgnoreNaNPayload=*/true);
        EXPECT_EQ(Res, Expected)
            << "Suboptimal result for " << LHS << " / " << RHS << ". Expected "
            << Expected << ", but got " << Res;
      },
      SparseLevel::SpecialValuesOnly);
#endif
}

TEST_F(ConstantFPRangeTest, flushDenormals) {
  const fltSemantics &FP8Sem = APFloat::Float8E4M3();
  APFloat NormalVal = APFloat::getSmallestNormalized(FP8Sem);
  APFloat Subnormal1 = NormalVal;
  Subnormal1.next(/*nextDown=*/true);
  APFloat Subnormal2 = APFloat::getSmallest(FP8Sem);
  APFloat ZeroVal = APFloat::getZero(FP8Sem);
  APFloat EdgeValues[8] = {-NormalVal, -Subnormal1, -Subnormal2, -ZeroVal,
                           ZeroVal,    Subnormal2,  Subnormal1,  NormalVal};
  constexpr DenormalMode::DenormalModeKind Modes[4] = {
      DenormalMode::IEEE, DenormalMode::PreserveSign,
      DenormalMode::PositiveZero, DenormalMode::Dynamic};
  for (uint32_t I = 0; I != 8; ++I) {
    for (uint32_t J = I; J != 8; ++J) {
      ConstantFPRange OriginCR =
          ConstantFPRange::getNonNaN(EdgeValues[I], EdgeValues[J]);
      for (auto Mode : Modes) {
        StringRef ModeName = denormalModeKindName(Mode);
        ConstantFPRange FlushedCR = OriginCR;
        FlushedCR.flushDenormals(Mode);

        ConstantFPRange Expected = ConstantFPRange::getEmpty(FP8Sem);
        auto CheckFlushedV = [&](const APFloat &V, const APFloat &FlushedV) {
          EXPECT_TRUE(FlushedCR.contains(FlushedV))
              << "Wrong result for flushDenormal(" << V << ", " << ModeName
              << "). The result " << FlushedCR << " should contain "
              << FlushedV;
          if (!Expected.contains(FlushedV))
            Expected = Expected.unionWith(ConstantFPRange(FlushedV));
        };
        EnumerateValuesInConstantFPRange(
            OriginCR,
            [&](const APFloat &V) {
              if (V.isDenormal()) {
                switch (Mode) {
                case DenormalMode::IEEE:
                  break;
                case DenormalMode::PreserveSign:
                  CheckFlushedV(V, APFloat::getZero(FP8Sem, V.isNegative()));
                  break;
                case DenormalMode::PositiveZero:
                  CheckFlushedV(V, APFloat::getZero(FP8Sem));
                  break;
                case DenormalMode::Dynamic:
                  // PreserveSign
                  CheckFlushedV(V, APFloat::getZero(FP8Sem, V.isNegative()));
                  // PositiveZero
                  CheckFlushedV(V, APFloat::getZero(FP8Sem));
                  break;
                default:
                  llvm_unreachable("unknown denormal mode");
                }
              }
              // It is not mandated that flushing to zero occurs.
              CheckFlushedV(V, V);
            },
            /*IgnoreNaNPayload=*/true);
        EXPECT_EQ(FlushedCR, Expected)
            << "Suboptimal result for flushDenormal(" << OriginCR << ", "
            << ModeName << "). Expected " << Expected << ", but got "
            << FlushedCR;
      }
    }
  }
}

} // anonymous namespace
