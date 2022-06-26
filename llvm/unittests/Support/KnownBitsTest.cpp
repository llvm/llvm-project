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

        KnownBits KnownComputed =
            KnownBits::computeForAddCarry(Known1, Known2, KnownCarry);
        EXPECT_EQ(Known, KnownComputed);
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

      KnownBits KnownComputed =
          KnownBits::computeForAddSub(IsAdd, /*NSW*/ false, Known1, Known2);
      EXPECT_EQ(Known, KnownComputed);

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
      KnownBits KnownMulHS(KnownAnd);
      KnownBits KnownMulHU(KnownAnd);
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

          Res = (N1.sext(2 * Bits) * N2.sext(2 * Bits)).extractBits(Bits, Bits);
          KnownMulHS.One &= Res;
          KnownMulHS.Zero &= ~Res;

          Res = (N1.zext(2 * Bits) * N2.zext(2 * Bits)).extractBits(Bits, Bits);
          KnownMulHU.One &= Res;
          KnownMulHU.Zero &= ~Res;

          if (!N2.isZero()) {
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
      EXPECT_EQ(KnownAnd, ComputedAnd);

      KnownBits ComputedOr = Known1 | Known2;
      EXPECT_EQ(KnownOr, ComputedOr);

      KnownBits ComputedXor = Known1 ^ Known2;
      EXPECT_EQ(KnownXor, ComputedXor);

      KnownBits ComputedUMax = KnownBits::umax(Known1, Known2);
      EXPECT_EQ(KnownUMax, ComputedUMax);

      KnownBits ComputedUMin = KnownBits::umin(Known1, Known2);
      EXPECT_EQ(KnownUMin, ComputedUMin);

      KnownBits ComputedSMax = KnownBits::smax(Known1, Known2);
      EXPECT_EQ(KnownSMax, ComputedSMax);

      KnownBits ComputedSMin = KnownBits::smin(Known1, Known2);
      EXPECT_EQ(KnownSMin, ComputedSMin);

      // The following are conservatively correct, but not guaranteed to be
      // precise.
      KnownBits ComputedMul = KnownBits::mul(Known1, Known2);
      EXPECT_TRUE(ComputedMul.Zero.isSubsetOf(KnownMul.Zero));
      EXPECT_TRUE(ComputedMul.One.isSubsetOf(KnownMul.One));

      KnownBits ComputedMulHS = KnownBits::mulhs(Known1, Known2);
      EXPECT_TRUE(ComputedMulHS.Zero.isSubsetOf(KnownMulHS.Zero));
      EXPECT_TRUE(ComputedMulHS.One.isSubsetOf(KnownMulHS.One));

      KnownBits ComputedMulHU = KnownBits::mulhu(Known1, Known2);
      EXPECT_TRUE(ComputedMulHU.Zero.isSubsetOf(KnownMulHU.Zero));
      EXPECT_TRUE(ComputedMulHU.One.isSubsetOf(KnownMulHU.One));

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

  // Also test 'unary' binary cases where the same argument is repeated.
  ForeachKnownBits(Bits, [&](const KnownBits &Known) {
    KnownBits KnownMul(Bits);
    KnownMul.Zero.setAllBits();
    KnownMul.One.setAllBits();

    ForeachNumInKnownBits(Known, [&](const APInt &N) {
      APInt Res = N * N;
      KnownMul.One &= Res;
      KnownMul.Zero &= ~Res;
    });

    KnownBits ComputedMul = KnownBits::mul(Known, Known, /*SelfMultiply*/ true);
    EXPECT_TRUE(ComputedMul.Zero.isSubsetOf(KnownMul.Zero));
    EXPECT_TRUE(ComputedMul.One.isSubsetOf(KnownMul.One));
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

TEST(KnownBitsTest, ICmpExhaustive) {
  unsigned Bits = 4;
  ForeachKnownBits(Bits, [&](const KnownBits &Known1) {
    ForeachKnownBits(Bits, [&](const KnownBits &Known2) {
      bool AllEQ = true, NoneEQ = true;
      bool AllNE = true, NoneNE = true;
      bool AllUGT = true, NoneUGT = true;
      bool AllUGE = true, NoneUGE = true;
      bool AllULT = true, NoneULT = true;
      bool AllULE = true, NoneULE = true;
      bool AllSGT = true, NoneSGT = true;
      bool AllSGE = true, NoneSGE = true;
      bool AllSLT = true, NoneSLT = true;
      bool AllSLE = true, NoneSLE = true;

      ForeachNumInKnownBits(Known1, [&](const APInt &N1) {
        ForeachNumInKnownBits(Known2, [&](const APInt &N2) {
          AllEQ &= N1.eq(N2);
          AllNE &= N1.ne(N2);
          AllUGT &= N1.ugt(N2);
          AllUGE &= N1.uge(N2);
          AllULT &= N1.ult(N2);
          AllULE &= N1.ule(N2);
          AllSGT &= N1.sgt(N2);
          AllSGE &= N1.sge(N2);
          AllSLT &= N1.slt(N2);
          AllSLE &= N1.sle(N2);
          NoneEQ &= !N1.eq(N2);
          NoneNE &= !N1.ne(N2);
          NoneUGT &= !N1.ugt(N2);
          NoneUGE &= !N1.uge(N2);
          NoneULT &= !N1.ult(N2);
          NoneULE &= !N1.ule(N2);
          NoneSGT &= !N1.sgt(N2);
          NoneSGE &= !N1.sge(N2);
          NoneSLT &= !N1.slt(N2);
          NoneSLE &= !N1.sle(N2);
        });
      });

      Optional<bool> KnownEQ = KnownBits::eq(Known1, Known2);
      Optional<bool> KnownNE = KnownBits::ne(Known1, Known2);
      Optional<bool> KnownUGT = KnownBits::ugt(Known1, Known2);
      Optional<bool> KnownUGE = KnownBits::uge(Known1, Known2);
      Optional<bool> KnownULT = KnownBits::ult(Known1, Known2);
      Optional<bool> KnownULE = KnownBits::ule(Known1, Known2);
      Optional<bool> KnownSGT = KnownBits::sgt(Known1, Known2);
      Optional<bool> KnownSGE = KnownBits::sge(Known1, Known2);
      Optional<bool> KnownSLT = KnownBits::slt(Known1, Known2);
      Optional<bool> KnownSLE = KnownBits::sle(Known1, Known2);

      EXPECT_EQ(AllEQ || NoneEQ, KnownEQ.has_value());
      EXPECT_EQ(AllNE || NoneNE, KnownNE.has_value());
      EXPECT_EQ(AllUGT || NoneUGT, KnownUGT.has_value());
      EXPECT_EQ(AllUGE || NoneUGE, KnownUGE.has_value());
      EXPECT_EQ(AllULT || NoneULT, KnownULT.has_value());
      EXPECT_EQ(AllULE || NoneULE, KnownULE.has_value());
      EXPECT_EQ(AllSGT || NoneSGT, KnownSGT.has_value());
      EXPECT_EQ(AllSGE || NoneSGE, KnownSGE.has_value());
      EXPECT_EQ(AllSLT || NoneSLT, KnownSLT.has_value());
      EXPECT_EQ(AllSLE || NoneSLE, KnownSLE.has_value());

      EXPECT_EQ(AllEQ, KnownEQ && *KnownEQ);
      EXPECT_EQ(AllNE, KnownNE && *KnownNE);
      EXPECT_EQ(AllUGT, KnownUGT && *KnownUGT);
      EXPECT_EQ(AllUGE, KnownUGE && *KnownUGE);
      EXPECT_EQ(AllULT, KnownULT && *KnownULT);
      EXPECT_EQ(AllULE, KnownULE && *KnownULE);
      EXPECT_EQ(AllSGT, KnownSGT && *KnownSGT);
      EXPECT_EQ(AllSGE, KnownSGE && *KnownSGE);
      EXPECT_EQ(AllSLT, KnownSLT && *KnownSLT);
      EXPECT_EQ(AllSLE, KnownSLE && *KnownSLE);

      EXPECT_EQ(NoneEQ, KnownEQ && !*KnownEQ);
      EXPECT_EQ(NoneNE, KnownNE && !*KnownNE);
      EXPECT_EQ(NoneUGT, KnownUGT && !*KnownUGT);
      EXPECT_EQ(NoneUGE, KnownUGE && !*KnownUGE);
      EXPECT_EQ(NoneULT, KnownULT && !*KnownULT);
      EXPECT_EQ(NoneULE, KnownULE && !*KnownULE);
      EXPECT_EQ(NoneSGT, KnownSGT && !*KnownSGT);
      EXPECT_EQ(NoneSGE, KnownSGE && !*KnownSGE);
      EXPECT_EQ(NoneSLT, KnownSLT && !*KnownSLT);
      EXPECT_EQ(NoneSLE, KnownSLE && !*KnownSLE);
    });
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

TEST(KnownBitsTest, GetSignedMinMaxVal) {
  unsigned Bits = 4;
  ForeachKnownBits(Bits, [&](const KnownBits &Known) {
    APInt Min = APInt::getSignedMaxValue(Bits);
    APInt Max = APInt::getSignedMinValue(Bits);
    ForeachNumInKnownBits(Known, [&](const APInt &N) {
      Min = APIntOps::smin(Min, N);
      Max = APIntOps::smax(Max, N);
    });
    EXPECT_EQ(Min, Known.getSignedMinValue());
    EXPECT_EQ(Max, Known.getSignedMaxValue());
  });
}

TEST(KnownBitsTest, CountMaxActiveBits) {
  unsigned Bits = 4;
  ForeachKnownBits(Bits, [&](const KnownBits &Known) {
    unsigned Expected = 0;
    ForeachNumInKnownBits(Known, [&](const APInt &N) {
      Expected = std::max(Expected, N.getActiveBits());
    });
    EXPECT_EQ(Expected, Known.countMaxActiveBits());
  });
}

TEST(KnownBitsTest, CountMaxSignificantBits) {
  unsigned Bits = 4;
  ForeachKnownBits(Bits, [&](const KnownBits &Known) {
    unsigned Expected = 0;
    ForeachNumInKnownBits(Known, [&](const APInt &N) {
      Expected = std::max(Expected, N.getSignificantBits());
    });
    EXPECT_EQ(Expected, Known.countMaxSignificantBits());
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
      EXPECT_EQ(Test, Baseline);
    }
  }
}

TEST(KnownBitsTest, SExtInReg) {
  unsigned Bits = 4;
  for (unsigned FromBits = 1; FromBits <= Bits; ++FromBits) {
    ForeachKnownBits(Bits, [&](const KnownBits &Known) {
      APInt CommonOne = APInt::getAllOnes(Bits);
      APInt CommonZero = APInt::getAllOnes(Bits);
      unsigned ExtBits = Bits - FromBits;
      ForeachNumInKnownBits(Known, [&](const APInt &N) {
        APInt Ext = N << ExtBits;
        Ext.ashrInPlace(ExtBits);
        CommonOne &= Ext;
        CommonZero &= ~Ext;
      });
      KnownBits KnownSExtInReg = Known.sextInReg(FromBits);
      EXPECT_EQ(CommonOne, KnownSExtInReg.One);
      EXPECT_EQ(CommonZero, KnownSExtInReg.Zero);
    });
  }
}

TEST(KnownBitsTest, CommonBitsSet) {
  unsigned Bits = 4;
  ForeachKnownBits(Bits, [&](const KnownBits &Known1) {
    ForeachKnownBits(Bits, [&](const KnownBits &Known2) {
      bool HasCommonBitsSet = false;
      ForeachNumInKnownBits(Known1, [&](const APInt &N1) {
        ForeachNumInKnownBits(Known2, [&](const APInt &N2) {
          HasCommonBitsSet |= N1.intersects(N2);
        });
      });
      EXPECT_EQ(!HasCommonBitsSet,
                KnownBits::haveNoCommonBitsSet(Known1, Known2));
    });
  });
}

} // end anonymous namespace
