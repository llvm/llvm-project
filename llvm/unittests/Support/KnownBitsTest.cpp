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
#include "gtest/gtest.h"

using namespace llvm;

namespace {

template<typename FnTy>
void ForeachKnownBits(unsigned Bits, FnTy Fn) {
  unsigned Max = 1 << Bits;
  KnownBits Known(Bits);
  for (unsigned Zero = 0; Zero < Max; ++Zero) {
    for (unsigned One = 0; One < Max; ++One) {
      Known.Zero = Zero;
      Known.One = One;
      if (Known.hasConflict())
        continue;

      Fn(Known);
    }
  }
}

template<typename FnTy>
void ForeachNumInKnownBits(const KnownBits &Known, FnTy Fn) {
  unsigned Bits = Known.getBitWidth();
  unsigned Max = 1 << Bits;
  for (unsigned N = 0; N < Max; ++N) {
    APInt Num(Bits, N);
    if ((Num & Known.Zero) != 0 || (~Num & Known.One) != 0)
      continue;

    Fn(Num);
  }
}

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

} // end anonymous namespace
