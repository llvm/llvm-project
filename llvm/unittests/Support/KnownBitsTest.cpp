//===- llvm/unittest/Support/KnownBitsTest.cpp - KnownBits tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for KnownBits functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/KnownBits.h"
#include "KnownBitsTest.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(KnownBitsTest, AddCarryExhaustive) {
  unsigned Bits = 4;
  ForeachKnownBits(Bits, [&](const KnownBits &Known1) {
    ForeachKnownBits(Bits, [&](const KnownBits &Known2) {
      ForeachKnownBits(1, [&](const KnownBits &KnownCarry) {
        // Explicitly compute known bits of the addition by trying all
        // possibilities.
        KnownBits Known(Bits);
        Known.Zero.setAllBits();
        Known.One.setAllBits();
        ForeachNumInKnownBits(Known1, [&](const APInt &N1) {
          ForeachNumInKnownBits(Known2, [&](const APInt &N2) {
            ForeachNumInKnownBits(KnownCarry, [&](const APInt &Carry) {
              APInt Add = N1 + N2;
              if (Carry.getBoolValue())
                ++Add;

              Known.One &= Add;
              Known.Zero &= ~Add;
            });
          });
        });

        KnownBits KnownComputed = KnownBits::computeForAddCarry(
            Known1, Known2, KnownCarry);
        EXPECT_EQ(Known.Zero, KnownComputed.Zero);
        EXPECT_EQ(Known.One, KnownComputed.One);
      });
    });
  });
}

static void TestAddSubExhaustive(bool IsAdd) {
  unsigned Bits = 4;
  ForeachKnownBits(Bits, [&](const KnownBits &Known1) {
    ForeachKnownBits(Bits, [&](const KnownBits &Known2) {
      KnownBits Known(Bits), KnownNSW(Bits);
      Known.Zero.setAllBits();
      Known.One.setAllBits();
      KnownNSW.Zero.setAllBits();
      KnownNSW.One.setAllBits();

      ForeachNumInKnownBits(Known1, [&](const APInt &N1) {
        ForeachNumInKnownBits(Known2, [&](const APInt &N2) {
          bool Overflow;
          APInt Res;
          if (IsAdd)
            Res = N1.sadd_ov(N2, Overflow);
          else
            Res = N1.ssub_ov(N2, Overflow);

          Known.One &= Res;
          Known.Zero &= ~Res;

          if (!Overflow) {
            KnownNSW.One &= Res;
            KnownNSW.Zero &= ~Res;
          }
        });
      });

      KnownBits KnownComputed = KnownBits::computeForAddSub(
          IsAdd, /*NSW*/false, Known1, Known2);
      EXPECT_EQ(Known.Zero, KnownComputed.Zero);
      EXPECT_EQ(Known.One, KnownComputed.One);

      // The NSW calculation is not precise, only check that it's
      // conservatively correct.
      KnownBits KnownNSWComputed = KnownBits::computeForAddSub(
          IsAdd, /*NSW*/true, Known1, Known2);
      EXPECT_TRUE(KnownNSWComputed.Zero.isSubsetOf(KnownNSW.Zero));
      EXPECT_TRUE(KnownNSWComputed.One.isSubsetOf(KnownNSW.One));
    });
  });
}

TEST(KnownBitsTest, AddSubExhaustive) {
  TestAddSubExhaustive(true);
  TestAddSubExhaustive(false);
}

TEST(KnownBitsTest, BinaryExhaustive) {
  unsigned Bits = 4;
  ForeachKnownBits(Bits, [&](const KnownBits &Known1) {
    ForeachKnownBits(Bits, [&](const KnownBits &Known2) {
      KnownBits KnownAnd(Bits);
      KnownAnd.Zero.setAllBits();
      KnownAnd.One.setAllBits();
      KnownBits KnownOr(KnownAnd);
      KnownBits KnownXor(KnownAnd);
      KnownBits KnownUMax(KnownAnd);
      KnownBits KnownUMin(KnownAnd);
      KnownBits KnownSMax(KnownAnd);
      KnownBits KnownSMin(KnownAnd);
      KnownBits KnownMul(KnownAnd);
      KnownBits KnownUDiv(KnownAnd);
      KnownBits KnownURem(KnownAnd);
      KnownBits KnownSRem(KnownAnd);
      KnownBits KnownShl(KnownAnd);
      KnownBits KnownLShr(KnownAnd);
      KnownBits KnownAShr(KnownAnd);

      ForeachNumInKnownBits(Known1, [&](const APInt &N1) {
        ForeachNumInKnownBits(Known2, [&](const APInt &N2) {
          APInt Res;

          Res = N1 & N2;
          KnownAnd.One &= Res;
          KnownAnd.Zero &= ~Res;

          Res = N1 | N2;
          KnownOr.One &= Res;
          KnownOr.Zero &= ~Res;

          Res = N1 ^ N2;
          KnownXor.One &= Res;
          KnownXor.Zero &= ~Res;

          Res = APIntOps::umax(N1, N2);
          KnownUMax.One &= Res;
          KnownUMax.Zero &= ~Res;

          Res = APIntOps::umin(N1, N2);
          KnownUMin.One &= Res;
          KnownUMin.Zero &= ~Res;

          Res = APIntOps::smax(N1, N2);
          KnownSMax.One &= Res;
          KnownSMax.Zero &= ~Res;

          Res = APIntOps::smin(N1, N2);
          KnownSMin.One &= Res;
          KnownSMin.Zero &= ~Res;

          Res = N1 * N2;
          KnownMul.One &= Res;
          KnownMul.Zero &= ~Res;

          if (!N2.isNullValue()) {
            Res = N1.udiv(N2);
            KnownUDiv.One &= Res;
            KnownUDiv.Zero &= ~Res;

            Res = N1.urem(N2);
            KnownURem.One &= Res;
            KnownURem.Zero &= ~Res;

            Res = N1.srem(N2);
            KnownSRem.One &= Res;
            KnownSRem.Zero &= ~Res;
          }

          if (N2.ult(1ULL << N1.getBitWidth())) {
            Res = N1.shl(N2);
            KnownShl.One &= Res;
            KnownShl.Zero &= ~Res;

            Res = N1.lshr(N2);
            KnownLShr.One &= Res;
            KnownLShr.Zero &= ~Res;

            Res = N1.ashr(N2);
            KnownAShr.One &= Res;
            KnownAShr.Zero &= ~Res;
          } else {
            KnownShl.resetAll();
            KnownLShr.resetAll();
            KnownAShr.resetAll();
          }
        });
      });

      KnownBits ComputedAnd = Known1 & Known2;
      EXPECT_EQ(KnownAnd.Zero, ComputedAnd.Zero);
      EXPECT_EQ(KnownAnd.One, ComputedAnd.One);

      KnownBits ComputedOr = Known1 | Known2;
      EXPECT_EQ(KnownOr.Zero, ComputedOr.Zero);
      EXPECT_EQ(KnownOr.One, ComputedOr.One);

      KnownBits ComputedXor = Known1 ^ Known2;
      EXPECT_EQ(KnownXor.Zero, ComputedXor.Zero);
      EXPECT_EQ(KnownXor.One, ComputedXor.One);

      KnownBits ComputedUMax = KnownBits::umax(Known1, Known2);
      EXPECT_EQ(KnownUMax.Zero, ComputedUMax.Zero);
      EXPECT_EQ(KnownUMax.One, ComputedUMax.One);

      KnownBits ComputedUMin = KnownBits::umin(Known1, Known2);
      EXPECT_EQ(KnownUMin.Zero, ComputedUMin.Zero);
      EXPECT_EQ(KnownUMin.One, ComputedUMin.One);

      KnownBits ComputedSMax = KnownBits::smax(Known1, Known2);
      EXPECT_EQ(KnownSMax.Zero, ComputedSMax.Zero);
      EXPECT_EQ(KnownSMax.One, ComputedSMax.One);

      KnownBits ComputedSMin = KnownBits::smin(Known1, Known2);
      EXPECT_EQ(KnownSMin.Zero, ComputedSMin.Zero);
      EXPECT_EQ(KnownSMin.One, ComputedSMin.One);

      // ComputedMul is conservatively correct, but not guaranteed to be
      // precise.
      KnownBits ComputedMul = KnownBits::computeForMul(Known1, Known2);
      EXPECT_TRUE(ComputedMul.Zero.isSubsetOf(KnownMul.Zero));
      EXPECT_TRUE(ComputedMul.One.isSubsetOf(KnownMul.One));

      KnownBits ComputedUDiv = KnownBits::udiv(Known1, Known2);
      EXPECT_TRUE(ComputedUDiv.Zero.isSubsetOf(KnownUDiv.Zero));
      EXPECT_TRUE(ComputedUDiv.One.isSubsetOf(KnownUDiv.One));

      KnownBits ComputedURem = KnownBits::urem(Known1, Known2);
      EXPECT_TRUE(ComputedURem.Zero.isSubsetOf(KnownURem.Zero));
      EXPECT_TRUE(ComputedURem.One.isSubsetOf(KnownURem.One));

      KnownBits ComputedSRem = KnownBits::srem(Known1, Known2);
      EXPECT_TRUE(ComputedSRem.Zero.isSubsetOf(KnownSRem.Zero));
      EXPECT_TRUE(ComputedSRem.One.isSubsetOf(KnownSRem.One));

      KnownBits ComputedShl = KnownBits::shl(Known1, Known2);
      EXPECT_TRUE(ComputedShl.Zero.isSubsetOf(KnownShl.Zero));
      EXPECT_TRUE(ComputedShl.One.isSubsetOf(KnownShl.One));

      KnownBits ComputedLShr = KnownBits::lshr(Known1, Known2);
      EXPECT_TRUE(ComputedLShr.Zero.isSubsetOf(KnownLShr.Zero));
      EXPECT_TRUE(ComputedLShr.One.isSubsetOf(KnownLShr.One));

      KnownBits ComputedAShr = KnownBits::ashr(Known1, Known2);
      EXPECT_TRUE(ComputedAShr.Zero.isSubsetOf(KnownAShr.Zero));
      EXPECT_TRUE(ComputedAShr.One.isSubsetOf(KnownAShr.One));
    });
  });
}

TEST(KnownBitsTest, UnaryExhaustive) {
  unsigned Bits = 4;
  ForeachKnownBits(Bits, [&](const KnownBits &Known) {
    KnownBits KnownAbs(Bits);
    KnownAbs.Zero.setAllBits();
    KnownAbs.One.setAllBits();
    KnownBits KnownAbsPoison(KnownAbs);

    ForeachNumInKnownBits(Known, [&](const APInt &N) {
      APInt Res = N.abs();
      KnownAbs.One &= Res;
      KnownAbs.Zero &= ~Res;

      if (!N.isMinSignedValue()) {
        KnownAbsPoison.One &= Res;
        KnownAbsPoison.Zero &= ~Res;
      }
    });

    // abs() is conservatively correct, but not guaranteed to be precise.
    KnownBits ComputedAbs = Known.abs();
    EXPECT_TRUE(ComputedAbs.Zero.isSubsetOf(KnownAbs.Zero));
    EXPECT_TRUE(ComputedAbs.One.isSubsetOf(KnownAbs.One));

    KnownBits ComputedAbsPoison = Known.abs(true);
    EXPECT_TRUE(ComputedAbsPoison.Zero.isSubsetOf(KnownAbsPoison.Zero));
    EXPECT_TRUE(ComputedAbsPoison.One.isSubsetOf(KnownAbsPoison.One));
  });
}

TEST(KnownBitsTest, GetMinMaxVal) {
  unsigned Bits = 4;
  ForeachKnownBits(Bits, [&](const KnownBits &Known) {
    APInt Min = APInt::getMaxValue(Bits);
    APInt Max = APInt::getMinValue(Bits);
    ForeachNumInKnownBits(Known, [&](const APInt &N) {
      Min = APIntOps::umin(Min, N);
      Max = APIntOps::umax(Max, N);
    });
    EXPECT_EQ(Min, Known.getMinValue());
    EXPECT_EQ(Max, Known.getMaxValue());
  });
}

TEST(KnownBitsTest, SExtOrTrunc) {
  const unsigned NarrowerSize = 4;
  const unsigned BaseSize = 6;
  const unsigned WiderSize = 8;
  APInt NegativeFitsNarrower(BaseSize, -4, /*isSigned*/ true);
  APInt NegativeDoesntFitNarrower(BaseSize, -28, /*isSigned*/ true);
  APInt PositiveFitsNarrower(BaseSize, 14);
  APInt PositiveDoesntFitNarrower(BaseSize, 36);
  auto InitKnownBits = [&](KnownBits &Res, const APInt &Input) {
    Res = KnownBits(Input.getBitWidth());
    Res.One = Input;
    Res.Zero = ~Input;
  };

  for (unsigned Size : {NarrowerSize, BaseSize, WiderSize}) {
    for (const APInt &Input :
         {NegativeFitsNarrower, NegativeDoesntFitNarrower, PositiveFitsNarrower,
          PositiveDoesntFitNarrower}) {
      KnownBits Test;
      InitKnownBits(Test, Input);
      KnownBits Baseline;
      InitKnownBits(Baseline, Input.sextOrTrunc(Size));
      Test = Test.sextOrTrunc(Size);
      EXPECT_EQ(Test.One, Baseline.One);
      EXPECT_EQ(Test.Zero, Baseline.Zero);
    }
  }
}

} // end anonymous namespace
