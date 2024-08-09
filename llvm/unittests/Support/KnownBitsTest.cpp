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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "gtest/gtest.h"

using namespace llvm;

using UnaryBitsFn = llvm::function_ref<KnownBits(const KnownBits &)>;
using UnaryIntFn = llvm::function_ref<std::optional<APInt>(const APInt &)>;

using BinaryBitsFn =
    llvm::function_ref<KnownBits(const KnownBits &, const KnownBits &)>;
using BinaryIntFn =
    llvm::function_ref<std::optional<APInt>(const APInt &, const APInt &)>;

static testing::AssertionResult checkResult(Twine Name, const KnownBits &Exact,
                                            const KnownBits &Computed,
                                            ArrayRef<KnownBits> Inputs,
                                            bool CheckOptimality) {
  if (CheckOptimality) {
    // We generally don't want to return conflicting known bits, even if it is
    // legal for always poison results.
    if (Exact.hasConflict() || Computed == Exact)
      return testing::AssertionSuccess();
  } else {
    if (Computed.Zero.isSubsetOf(Exact.Zero) &&
        Computed.One.isSubsetOf(Exact.One))
      return testing::AssertionSuccess();
  }

  testing::AssertionResult Result = testing::AssertionFailure();
  Result << Name << ": ";
  Result << "Inputs = ";
  for (const KnownBits &Input : Inputs)
    Result << Input << ", ";
  Result << "Computed = " << Computed << ", Exact = " << Exact;
  return Result;
}

static void testUnaryOpExhaustive(StringRef Name, UnaryBitsFn BitsFn,
                                  UnaryIntFn IntFn,
                                  bool CheckOptimality = true) {
  for (unsigned Bits : {1, 4}) {
    ForeachKnownBits(Bits, [&](const KnownBits &Known) {
      KnownBits Computed = BitsFn(Known);
      KnownBits Exact(Bits);
      Exact.Zero.setAllBits();
      Exact.One.setAllBits();

      ForeachNumInKnownBits(Known, [&](const APInt &N) {
        if (std::optional<APInt> Res = IntFn(N)) {
          Exact.One &= *Res;
          Exact.Zero &= ~*Res;
        }
      });

      if (!Exact.hasConflict()) {
        EXPECT_TRUE(checkResult(Name, Exact, Computed, Known, CheckOptimality));
      }
    });
  }
}

static void testBinaryOpExhaustive(StringRef Name, BinaryBitsFn BitsFn,
                                   BinaryIntFn IntFn,
                                   bool CheckOptimality = true,
                                   bool RefinePoisonToZero = false) {
  for (unsigned Bits : {1, 4}) {
    ForeachKnownBits(Bits, [&](const KnownBits &Known1) {
      ForeachKnownBits(Bits, [&](const KnownBits &Known2) {
        KnownBits Computed = BitsFn(Known1, Known2);
        KnownBits Exact(Bits);
        Exact.Zero.setAllBits();
        Exact.One.setAllBits();

        ForeachNumInKnownBits(Known1, [&](const APInt &N1) {
          ForeachNumInKnownBits(Known2, [&](const APInt &N2) {
            if (std::optional<APInt> Res = IntFn(N1, N2)) {
              Exact.One &= *Res;
              Exact.Zero &= ~*Res;
            }
          });
        });

        if (!Exact.hasConflict()) {
          EXPECT_TRUE(checkResult(Name, Exact, Computed, {Known1, Known2},
                                  CheckOptimality));
        }
        // In some cases we choose to return zero if the result is always
        // poison.
        if (RefinePoisonToZero && Exact.hasConflict() &&
            !Known1.hasConflict() && !Known2.hasConflict()) {
          EXPECT_TRUE(Computed.isZero());
        }
      });
    });
  }
}

namespace {

TEST(KnownBitsTest, AddCarryExhaustive) {
  unsigned Bits = 4;
  ForeachKnownBits(Bits, [&](const KnownBits &Known1) {
    ForeachKnownBits(Bits, [&](const KnownBits &Known2) {
      ForeachKnownBits(1, [&](const KnownBits &KnownCarry) {
        // Explicitly compute known bits of the addition by trying all
        // possibilities.
        KnownBits Exact(Bits);
        Exact.Zero.setAllBits();
        Exact.One.setAllBits();
        ForeachNumInKnownBits(Known1, [&](const APInt &N1) {
          ForeachNumInKnownBits(Known2, [&](const APInt &N2) {
            ForeachNumInKnownBits(KnownCarry, [&](const APInt &Carry) {
              APInt Add = N1 + N2;
              if (Carry.getBoolValue())
                ++Add;

              Exact.One &= Add;
              Exact.Zero &= ~Add;
            });
          });
        });

        KnownBits Computed =
            KnownBits::computeForAddCarry(Known1, Known2, KnownCarry);
        if (!Exact.hasConflict()) {
          EXPECT_EQ(Exact, Computed);
        }
      });
    });
  });
}

static void TestAddSubExhaustive(bool IsAdd) {
  Twine Name = IsAdd ? "add" : "sub";
  unsigned Bits = 4;
  ForeachKnownBits(Bits, [&](const KnownBits &Known1) {
    ForeachKnownBits(Bits, [&](const KnownBits &Known2) {
      KnownBits Exact(Bits), ExactNSW(Bits), ExactNUW(Bits),
          ExactNSWAndNUW(Bits);
      Exact.Zero.setAllBits();
      Exact.One.setAllBits();
      ExactNSW.Zero.setAllBits();
      ExactNSW.One.setAllBits();
      ExactNUW.Zero.setAllBits();
      ExactNUW.One.setAllBits();
      ExactNSWAndNUW.Zero.setAllBits();
      ExactNSWAndNUW.One.setAllBits();

      ForeachNumInKnownBits(Known1, [&](const APInt &N1) {
        ForeachNumInKnownBits(Known2, [&](const APInt &N2) {
          bool SignedOverflow;
          bool UnsignedOverflow;
          APInt Res;
          if (IsAdd) {
            Res = N1.uadd_ov(N2, UnsignedOverflow);
            Res = N1.sadd_ov(N2, SignedOverflow);
          } else {
            Res = N1.usub_ov(N2, UnsignedOverflow);
            Res = N1.ssub_ov(N2, SignedOverflow);
          }

          Exact.One &= Res;
          Exact.Zero &= ~Res;

          if (!SignedOverflow) {
            ExactNSW.One &= Res;
            ExactNSW.Zero &= ~Res;
          }

          if (!UnsignedOverflow) {
            ExactNUW.One &= Res;
            ExactNUW.Zero &= ~Res;
          }

          if (!UnsignedOverflow && !SignedOverflow) {
            ExactNSWAndNUW.One &= Res;
            ExactNSWAndNUW.Zero &= ~Res;
          }
        });
      });

      KnownBits Computed = KnownBits::computeForAddSub(
          IsAdd, /*NSW=*/false, /*NUW=*/false, Known1, Known2);
      EXPECT_TRUE(checkResult(Name, Exact, Computed, {Known1, Known2},
                              /*CheckOptimality=*/true));

      KnownBits ComputedNSW = KnownBits::computeForAddSub(
          IsAdd, /*NSW=*/true, /*NUW=*/false, Known1, Known2);
      EXPECT_TRUE(checkResult(Name + " nsw", ExactNSW, ComputedNSW,
                              {Known1, Known2},
                              /*CheckOptimality=*/true));

      KnownBits ComputedNUW = KnownBits::computeForAddSub(
          IsAdd, /*NSW=*/false, /*NUW=*/true, Known1, Known2);
      EXPECT_TRUE(checkResult(Name + " nuw", ExactNUW, ComputedNUW,
                              {Known1, Known2},
                              /*CheckOptimality=*/true));

      KnownBits ComputedNSWAndNUW = KnownBits::computeForAddSub(
          IsAdd, /*NSW=*/true, /*NUW=*/true, Known1, Known2);
      EXPECT_TRUE(checkResult(Name + " nsw nuw", ExactNSWAndNUW,
                              ComputedNSWAndNUW, {Known1, Known2},
                              /*CheckOptimality=*/true));
    });
  });
}

TEST(KnownBitsTest, AddSubExhaustive) {
  TestAddSubExhaustive(true);
  TestAddSubExhaustive(false);
}

TEST(KnownBitsTest, SubBorrowExhaustive) {
  unsigned Bits = 4;
  ForeachKnownBits(Bits, [&](const KnownBits &Known1) {
    ForeachKnownBits(Bits, [&](const KnownBits &Known2) {
      ForeachKnownBits(1, [&](const KnownBits &KnownBorrow) {
        // Explicitly compute known bits of the subtraction by trying all
        // possibilities.
        KnownBits Exact(Bits);
        Exact.Zero.setAllBits();
        Exact.One.setAllBits();
        ForeachNumInKnownBits(Known1, [&](const APInt &N1) {
          ForeachNumInKnownBits(Known2, [&](const APInt &N2) {
            ForeachNumInKnownBits(KnownBorrow, [&](const APInt &Borrow) {
              APInt Sub = N1 - N2;
              if (Borrow.getBoolValue())
                --Sub;

              Exact.One &= Sub;
              Exact.Zero &= ~Sub;
            });
          });
        });

        KnownBits Computed =
            KnownBits::computeForSubBorrow(Known1, Known2, KnownBorrow);
        if (!Exact.hasConflict()) {
          EXPECT_EQ(Exact, Computed);
        }
      });
    });
  });
}

TEST(KnownBitsTest, SignBitUnknown) {
  KnownBits Known(2);
  EXPECT_TRUE(Known.isSignUnknown());
  Known.Zero.setBit(0);
  EXPECT_TRUE(Known.isSignUnknown());
  Known.Zero.setBit(1);
  EXPECT_FALSE(Known.isSignUnknown());
  Known.Zero.clearBit(0);
  EXPECT_FALSE(Known.isSignUnknown());
  Known.Zero.clearBit(1);
  EXPECT_TRUE(Known.isSignUnknown());

  Known.One.setBit(0);
  EXPECT_TRUE(Known.isSignUnknown());
  Known.One.setBit(1);
  EXPECT_FALSE(Known.isSignUnknown());
  Known.One.clearBit(0);
  EXPECT_FALSE(Known.isSignUnknown());
  Known.One.clearBit(1);
  EXPECT_TRUE(Known.isSignUnknown());
}

TEST(KnownBitsTest, BinaryExhaustive) {
  testBinaryOpExhaustive(
      "and",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return Known1 & Known2;
      },
      [](const APInt &N1, const APInt &N2) { return N1 & N2; });
  testBinaryOpExhaustive(
      "or",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return Known1 | Known2;
      },
      [](const APInt &N1, const APInt &N2) { return N1 | N2; });
  testBinaryOpExhaustive(
      "xor",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return Known1 ^ Known2;
      },
      [](const APInt &N1, const APInt &N2) { return N1 ^ N2; });
  testBinaryOpExhaustive(
      "add",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return KnownBits::add(Known1, Known2);
      },
      [](const APInt &N1, const APInt &N2) { return N1 + N2; },
      /*CheckOptimality=*/false);
  testBinaryOpExhaustive(
      "sub",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return KnownBits::sub(Known1, Known2);
      },
      [](const APInt &N1, const APInt &N2) { return N1 - N2; },
      /*CheckOptimality=*/false);
  testBinaryOpExhaustive("umax", KnownBits::umax, APIntOps::umax);
  testBinaryOpExhaustive("umin", KnownBits::umin, APIntOps::umin);
  testBinaryOpExhaustive("smax", KnownBits::smax, APIntOps::smax);
  testBinaryOpExhaustive("smin", KnownBits::smin, APIntOps::smin);
  testBinaryOpExhaustive("abdu", KnownBits::abdu, APIntOps::abdu);
  testBinaryOpExhaustive("abds", KnownBits::abds, APIntOps::abds);
  testBinaryOpExhaustive(
      "udiv",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return KnownBits::udiv(Known1, Known2);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        if (N2.isZero())
          return std::nullopt;
        return N1.udiv(N2);
      },
      /*CheckOptimality=*/false);
  testBinaryOpExhaustive(
      "udiv exact",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return KnownBits::udiv(Known1, Known2, /*Exact=*/true);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        if (N2.isZero() || !N1.urem(N2).isZero())
          return std::nullopt;
        return N1.udiv(N2);
      },
      /*CheckOptimality=*/false);
  testBinaryOpExhaustive(
      "sdiv",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return KnownBits::sdiv(Known1, Known2);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        if (N2.isZero() || (N1.isMinSignedValue() && N2.isAllOnes()))
          return std::nullopt;
        return N1.sdiv(N2);
      },
      /*CheckOptimality=*/false);
  testBinaryOpExhaustive(
      "sdiv exact",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return KnownBits::sdiv(Known1, Known2, /*Exact=*/true);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        if (N2.isZero() || (N1.isMinSignedValue() && N2.isAllOnes()) ||
            !N1.srem(N2).isZero())
          return std::nullopt;
        return N1.sdiv(N2);
      },
      /*CheckOptimality=*/false);
  testBinaryOpExhaustive(
      "urem", KnownBits::urem,
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        if (N2.isZero())
          return std::nullopt;
        return N1.urem(N2);
      },
      /*CheckOptimality=*/false);
  testBinaryOpExhaustive(
      "srem", KnownBits::srem,
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        if (N2.isZero())
          return std::nullopt;
        return N1.srem(N2);
      },
      /*CheckOptimality=*/false);
  testBinaryOpExhaustive(
      "sadd_sat", KnownBits::sadd_sat,
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        return N1.sadd_sat(N2);
      },
      /*CheckOptimality=*/false);
  testBinaryOpExhaustive(
      "uadd_sat", KnownBits::uadd_sat,
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        return N1.uadd_sat(N2);
      },
      /*CheckOptimality=*/false);
  testBinaryOpExhaustive(
      "ssub_sat", KnownBits::ssub_sat,
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        return N1.ssub_sat(N2);
      },
      /*CheckOptimality=*/false);
  testBinaryOpExhaustive(
      "usub_sat", KnownBits::usub_sat,
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        return N1.usub_sat(N2);
      },
      /*CheckOptimality=*/false);
  testBinaryOpExhaustive(
      "shl",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return KnownBits::shl(Known1, Known2);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        if (N2.uge(N2.getBitWidth()))
          return std::nullopt;
        return N1.shl(N2);
      },
      /*CheckOptimality=*/true, /*RefinePoisonToZero=*/true);
  testBinaryOpExhaustive(
      "ushl_ov",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return KnownBits::shl(Known1, Known2, /*NUW=*/true);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        bool Overflow;
        APInt Res = N1.ushl_ov(N2, Overflow);
        if (Overflow)
          return std::nullopt;
        return Res;
      },
      /*CheckOptimality=*/true, /*RefinePoisonToZero=*/true);
  testBinaryOpExhaustive(
      "shl nsw",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return KnownBits::shl(Known1, Known2, /*NUW=*/false, /*NSW=*/true);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        bool Overflow;
        APInt Res = N1.sshl_ov(N2, Overflow);
        if (Overflow)
          return std::nullopt;
        return Res;
      },
      /*CheckOptimality=*/true, /*RefinePoisonToZero=*/true);
  testBinaryOpExhaustive(
      "shl nuw",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return KnownBits::shl(Known1, Known2, /*NUW=*/true, /*NSW=*/true);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        bool OverflowUnsigned, OverflowSigned;
        APInt Res = N1.ushl_ov(N2, OverflowUnsigned);
        (void)N1.sshl_ov(N2, OverflowSigned);
        if (OverflowUnsigned || OverflowSigned)
          return std::nullopt;
        return Res;
      },
      /*CheckOptimality=*/true, /*RefinePoisonToZero=*/true);

  testBinaryOpExhaustive(
      "lshr",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return KnownBits::lshr(Known1, Known2);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        if (N2.uge(N2.getBitWidth()))
          return std::nullopt;
        return N1.lshr(N2);
      },
      /*CheckOptimality=*/true, /*RefinePoisonToZero=*/true);
  testBinaryOpExhaustive(
      "lshr exact",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return KnownBits::lshr(Known1, Known2, /*ShAmtNonZero=*/false,
                               /*Exact=*/true);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        if (N2.uge(N2.getBitWidth()))
          return std::nullopt;
        if (!N1.extractBits(N2.getZExtValue(), 0).isZero())
          return std::nullopt;
        return N1.lshr(N2);
      },
      /*CheckOptimality=*/true, /*RefinePoisonToZero=*/true);
  testBinaryOpExhaustive(
      "ashr",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return KnownBits::ashr(Known1, Known2);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        if (N2.uge(N2.getBitWidth()))
          return std::nullopt;
        return N1.ashr(N2);
      },
      /*CheckOptimality=*/true, /*RefinePoisonToZero=*/true);
  testBinaryOpExhaustive(
      "ashr exact",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return KnownBits::ashr(Known1, Known2, /*ShAmtNonZero=*/false,
                               /*Exact=*/true);
      },
      [](const APInt &N1, const APInt &N2) -> std::optional<APInt> {
        if (N2.uge(N2.getBitWidth()))
          return std::nullopt;
        if (!N1.extractBits(N2.getZExtValue(), 0).isZero())
          return std::nullopt;
        return N1.ashr(N2);
      },
      /*CheckOptimality=*/true, /*RefinePoisonToZero=*/true);
  testBinaryOpExhaustive(
      "mul",
      [](const KnownBits &Known1, const KnownBits &Known2) {
        return KnownBits::mul(Known1, Known2);
      },
      [](const APInt &N1, const APInt &N2) { return N1 * N2; },
      /*CheckOptimality=*/false);
  testBinaryOpExhaustive(
      "mulhs", KnownBits::mulhs,
      [](const APInt &N1, const APInt &N2) { return APIntOps::mulhs(N1, N2); },
      /*CheckOptimality=*/false);
  testBinaryOpExhaustive(
      "mulhu", KnownBits::mulhu,
      [](const APInt &N1, const APInt &N2) { return APIntOps::mulhu(N1, N2); },
      /*CheckOptimality=*/false);

  testBinaryOpExhaustive("avgFloorS", KnownBits::avgFloorS, APIntOps::avgFloorS,
                         false);

  testBinaryOpExhaustive("avgFloorU", KnownBits::avgFloorU, APIntOps::avgFloorU,
                         false);

  testBinaryOpExhaustive("avgCeilU", KnownBits::avgCeilU, APIntOps::avgCeilU,
                         false);

  testBinaryOpExhaustive("avgCeilS", KnownBits::avgCeilS, APIntOps::avgCeilS,
                         false);
}

TEST(KnownBitsTest, UnaryExhaustive) {
  testUnaryOpExhaustive(
      "abs", [](const KnownBits &Known) { return Known.abs(); },
      [](const APInt &N) { return N.abs(); });

  testUnaryOpExhaustive(
      "abs(true)", [](const KnownBits &Known) { return Known.abs(true); },
      [](const APInt &N) -> std::optional<APInt> {
        if (N.isMinSignedValue())
          return std::nullopt;
        return N.abs();
      });

  testUnaryOpExhaustive(
      "blsi", [](const KnownBits &Known) { return Known.blsi(); },
      [](const APInt &N) { return N & -N; });
  testUnaryOpExhaustive(
      "blsmsk", [](const KnownBits &Known) { return Known.blsmsk(); },
      [](const APInt &N) { return N ^ (N - 1); });

  testUnaryOpExhaustive(
      "mul self",
      [](const KnownBits &Known) {
        return KnownBits::mul(Known, Known, /*SelfMultiply=*/true);
      },
      [](const APInt &N) { return N * N; }, /*CheckOptimality=*/false);
}

TEST(KnownBitsTest, WideShifts) {
  unsigned BitWidth = 128;
  KnownBits Unknown(BitWidth);
  KnownBits AllOnes = KnownBits::makeConstant(APInt::getAllOnes(BitWidth));

  KnownBits ShlResult(BitWidth);
  ShlResult.makeNegative();
  EXPECT_EQ(KnownBits::shl(AllOnes, Unknown), ShlResult);
  KnownBits LShrResult(BitWidth);
  LShrResult.One.setBit(0);
  EXPECT_EQ(KnownBits::lshr(AllOnes, Unknown), LShrResult);
  EXPECT_EQ(KnownBits::ashr(AllOnes, Unknown), AllOnes);
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

      std::optional<bool> KnownEQ = KnownBits::eq(Known1, Known2);
      std::optional<bool> KnownNE = KnownBits::ne(Known1, Known2);
      std::optional<bool> KnownUGT = KnownBits::ugt(Known1, Known2);
      std::optional<bool> KnownUGE = KnownBits::uge(Known1, Known2);
      std::optional<bool> KnownULT = KnownBits::ult(Known1, Known2);
      std::optional<bool> KnownULE = KnownBits::ule(Known1, Known2);
      std::optional<bool> KnownSGT = KnownBits::sgt(Known1, Known2);
      std::optional<bool> KnownSGE = KnownBits::sge(Known1, Known2);
      std::optional<bool> KnownSLT = KnownBits::slt(Known1, Known2);
      std::optional<bool> KnownSLE = KnownBits::sle(Known1, Known2);

      if (Known1.hasConflict() || Known2.hasConflict())
        return;

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

      EXPECT_EQ(AllEQ, KnownEQ.has_value() && *KnownEQ);
      EXPECT_EQ(AllNE, KnownNE.has_value() && *KnownNE);
      EXPECT_EQ(AllUGT, KnownUGT.has_value() && *KnownUGT);
      EXPECT_EQ(AllUGE, KnownUGE.has_value() && *KnownUGE);
      EXPECT_EQ(AllULT, KnownULT.has_value() && *KnownULT);
      EXPECT_EQ(AllULE, KnownULE.has_value() && *KnownULE);
      EXPECT_EQ(AllSGT, KnownSGT.has_value() && *KnownSGT);
      EXPECT_EQ(AllSGE, KnownSGE.has_value() && *KnownSGE);
      EXPECT_EQ(AllSLT, KnownSLT.has_value() && *KnownSLT);
      EXPECT_EQ(AllSLE, KnownSLE.has_value() && *KnownSLE);

      EXPECT_EQ(NoneEQ, KnownEQ.has_value() && !*KnownEQ);
      EXPECT_EQ(NoneNE, KnownNE.has_value() && !*KnownNE);
      EXPECT_EQ(NoneUGT, KnownUGT.has_value() && !*KnownUGT);
      EXPECT_EQ(NoneUGE, KnownUGE.has_value() && !*KnownUGE);
      EXPECT_EQ(NoneULT, KnownULT.has_value() && !*KnownULT);
      EXPECT_EQ(NoneULE, KnownULE.has_value() && !*KnownULE);
      EXPECT_EQ(NoneSGT, KnownSGT.has_value() && !*KnownSGT);
      EXPECT_EQ(NoneSGE, KnownSGE.has_value() && !*KnownSGE);
      EXPECT_EQ(NoneSLT, KnownSLT.has_value() && !*KnownSLT);
      EXPECT_EQ(NoneSLE, KnownSLE.has_value() && !*KnownSLE);
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
    if (!Known.hasConflict()) {
      EXPECT_EQ(Min, Known.getMinValue());
      EXPECT_EQ(Max, Known.getMaxValue());
    }
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
    if (!Known.hasConflict()) {
      EXPECT_EQ(Min, Known.getSignedMinValue());
      EXPECT_EQ(Max, Known.getSignedMaxValue());
    }
  });
}

TEST(KnownBitsTest, CountMaxActiveBits) {
  unsigned Bits = 4;
  ForeachKnownBits(Bits, [&](const KnownBits &Known) {
    unsigned Expected = 0;
    ForeachNumInKnownBits(Known, [&](const APInt &N) {
      Expected = std::max(Expected, N.getActiveBits());
    });
    if (!Known.hasConflict()) {
      EXPECT_EQ(Expected, Known.countMaxActiveBits());
    }
  });
}

TEST(KnownBitsTest, CountMaxSignificantBits) {
  unsigned Bits = 4;
  ForeachKnownBits(Bits, [&](const KnownBits &Known) {
    unsigned Expected = 0;
    ForeachNumInKnownBits(Known, [&](const APInt &N) {
      Expected = std::max(Expected, N.getSignificantBits());
    });
    if (!Known.hasConflict()) {
      EXPECT_EQ(Expected, Known.countMaxSignificantBits());
    }
  });
}

TEST(KnownBitsTest, SExtOrTrunc) {
  const unsigned NarrowerSize = 4;
  const unsigned BaseSize = 6;
  const unsigned WiderSize = 8;
  APInt NegativeFitsNarrower(BaseSize, -4, /*isSigned=*/true);
  APInt NegativeDoesntFitNarrower(BaseSize, -28, /*isSigned=*/true);
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
      if (!Known.hasConflict()) {
        EXPECT_EQ(CommonOne, KnownSExtInReg.One);
        EXPECT_EQ(CommonZero, KnownSExtInReg.Zero);
      }
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
      if (!Known1.hasConflict() && !Known2.hasConflict()) {
        EXPECT_EQ(!HasCommonBitsSet,
                  KnownBits::haveNoCommonBitsSet(Known1, Known2));
      }
    });
  });
}

TEST(KnownBitsTest, ConcatBits) {
  unsigned Bits = 4;
  for (unsigned LoBits = 1; LoBits < Bits; ++LoBits) {
    unsigned HiBits = Bits - LoBits;
    ForeachKnownBits(LoBits, [&](const KnownBits &KnownLo) {
      ForeachKnownBits(HiBits, [&](const KnownBits &KnownHi) {
        KnownBits KnownAll = KnownHi.concat(KnownLo);

        EXPECT_EQ(KnownLo.countMinPopulation() + KnownHi.countMinPopulation(),
                  KnownAll.countMinPopulation());
        EXPECT_EQ(KnownLo.countMaxPopulation() + KnownHi.countMaxPopulation(),
                  KnownAll.countMaxPopulation());

        KnownBits ExtractLo = KnownAll.extractBits(LoBits, 0);
        KnownBits ExtractHi = KnownAll.extractBits(HiBits, LoBits);

        EXPECT_EQ(KnownLo.One.getZExtValue(), ExtractLo.One.getZExtValue());
        EXPECT_EQ(KnownHi.One.getZExtValue(), ExtractHi.One.getZExtValue());
        EXPECT_EQ(KnownLo.Zero.getZExtValue(), ExtractLo.Zero.getZExtValue());
        EXPECT_EQ(KnownHi.Zero.getZExtValue(), ExtractHi.Zero.getZExtValue());
      });
    });
  }
}

} // end anonymous namespace
