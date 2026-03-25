//===- llvm/unittest/Support/DivisionByConstantTest.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APInt.h"
#include "llvm/Support/DivisionByConstantInfo.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

template <typename Fn> static void EnumerateAPInts(unsigned Bits, Fn TestFn) {
  APInt N(Bits, 0);
  do {
    TestFn(N);
  } while (++N != 0);
}

APInt MULHS(APInt X, APInt Y) {
  unsigned Bits = X.getBitWidth();
  unsigned WideBits = 2 * Bits;
  return (X.sext(WideBits) * Y.sext(WideBits)).lshr(Bits).trunc(Bits);
}

APInt SignedDivideUsingMagic(APInt Numerator, APInt Divisor,
                             SignedDivisionByConstantInfo Magics) {
  unsigned Bits = Numerator.getBitWidth();

  APInt Factor(Bits, 0);
  APInt ShiftMask(Bits, -1, true);
  if (Divisor.isOne() || Divisor.isAllOnes()) {
    // If d is +1/-1, we just multiply the numerator by +1/-1.
    Factor = Divisor.getSExtValue();
    Magics.Magic = 0;
    Magics.ShiftAmount = 0;
    ShiftMask = 0;
  } else if (Divisor.isStrictlyPositive() && Magics.Magic.isNegative()) {
    // If d > 0 and m < 0, add the numerator.
    Factor = 1;
  } else if (Divisor.isNegative() && Magics.Magic.isStrictlyPositive()) {
    // If d < 0 and m > 0, subtract the numerator.
    Factor = -1;
  }

  // Multiply the numerator by the magic value.
  APInt Q = MULHS(Numerator, Magics.Magic);

  // (Optionally) Add/subtract the numerator using Factor.
  Factor = Numerator * Factor;
  Q = Q + Factor;

  // Shift right algebraic by shift value.
  Q = Q.ashr(Magics.ShiftAmount);

  // Extract the sign bit, mask it and add it to the quotient.
  unsigned SignShift = Bits - 1;
  APInt T = Q.lshr(SignShift);
  T = T & ShiftMask;
  return Q + T;
}

TEST(SignedDivisionByConstantTest, Test) {
  for (unsigned Bits = 1; Bits <= 32; ++Bits) {
    if (Bits < 3)
      continue; // Not supported by `SignedDivisionByConstantInfo::get()`.
    if (Bits > 12)
      continue; // Unreasonably slow.
    EnumerateAPInts(Bits, [Bits](const APInt &Divisor) {
      if (Divisor.isZero())
        return; // Division by zero is undefined behavior.
      SignedDivisionByConstantInfo Magics;
      if (!(Divisor.isOne() || Divisor.isAllOnes()))
        Magics = SignedDivisionByConstantInfo::get(Divisor);
      EnumerateAPInts(Bits, [Divisor, Magics, Bits](const APInt &Numerator) {
        if (Numerator.isMinSignedValue() && Divisor.isAllOnes())
          return; // Overflow is undefined behavior.
        APInt NativeResult = Numerator.sdiv(Divisor);
        APInt MagicResult = SignedDivideUsingMagic(Numerator, Divisor, Magics);
        ASSERT_EQ(MagicResult, NativeResult)
            << " ... given the operation:  srem i" << Bits << " " << Numerator
            << ", " << Divisor;
      });
    });
  }
}

APInt MULHU(APInt X, APInt Y) {
  unsigned Bits = X.getBitWidth();
  unsigned WideBits = 2 * Bits;
  return (X.zext(WideBits) * Y.zext(WideBits)).lshr(Bits).trunc(Bits);
}

APInt WideMULHU(APInt X, APInt Y) {
  assert(X.getBitWidth() == Y.getBitWidth() && "Expected matching widths");
  unsigned Bits = X.getBitWidth();
  unsigned WideBits = 2 * Bits;
  return (X.zext(WideBits) * Y.zext(WideBits)).lshr(Bits).trunc(Bits);
}

APInt UnsignedDivideUsingMagic(const APInt &Numerator, const APInt &Divisor,
                               bool LZOptimization,
                               bool AllowEvenDivisorOptimization, bool ForceNPQ,
                               UnsignedDivisionByConstantInfo Magics) {
  assert(!Divisor.isOne() && "Division by 1 is not supported using Magic.");

  unsigned Bits = Numerator.getBitWidth();

  if (LZOptimization) {
    unsigned LeadingZeros = Numerator.countl_zero();
    // Clip to the number of leading zeros in the divisor.
    LeadingZeros = std::min(LeadingZeros, Divisor.countl_zero());
    if (LeadingZeros > 0) {
      Magics = UnsignedDivisionByConstantInfo::get(
          Divisor, LeadingZeros, AllowEvenDivisorOptimization);
      assert(!Magics.IsAdd && "Should use cheap fixup now");
    }
  }

  assert(Magics.PreShift < Divisor.getBitWidth() &&
         "We shouldn't generate an undefined shift!");
  assert((Magics.Widening != UnsignedDivisionByConstantWidening::FullMultiply ||
          Magics.PostShift < Magics.Magic.getBitWidth()) &&
         "We shouldn't generate an undefined shift!");
  assert((!Magics.IsAdd || Magics.PreShift == 0) && "Unexpected pre-shift");
  unsigned PreShift = Magics.PreShift;
  unsigned PostShift = Magics.PostShift;
  bool UseNPQ = Magics.IsAdd;

  if (Magics.Widening == UnsignedDivisionByConstantWidening::MulHigh) {
    unsigned WideBits = Magics.Magic.getBitWidth();
    APInt Q = WideMULHU(Numerator.zext(WideBits), Magics.Magic);
    return Q.trunc(Bits);
  }

  if (Magics.Widening == UnsignedDivisionByConstantWidening::FullMultiply) {
    unsigned WideBits = Magics.Magic.getBitWidth();
    APInt Q = Numerator.zext(WideBits) * Magics.Magic;
    return Q.lshr(PostShift).trunc(Bits);
  }

  APInt NPQFactor =
      UseNPQ ? APInt::getSignedMinValue(Bits) : APInt::getZero(Bits);

  APInt Q = Numerator.lshr(PreShift);

  // Multiply the numerator by the magic value.
  Q = MULHU(Q, Magics.Magic);

  if (UseNPQ || ForceNPQ) {
    APInt NPQ = Numerator - Q;

    // For vectors we might have a mix of non-NPQ/NPQ paths, so use
    // MULHU to act as a SRL-by-1 for NPQ, else multiply by zero.
    APInt NPQ_Scalar = NPQ.lshr(1);
    (void)NPQ_Scalar;
    NPQ = MULHU(NPQ, NPQFactor);
    assert(!UseNPQ || NPQ == NPQ_Scalar);

    Q = NPQ + Q;
  }

  Q = Q.lshr(PostShift);

  return Q;
}

TEST(UnsignedDivisionByConstantTest, Test) {
  for (unsigned Bits = 1; Bits <= 32; ++Bits) {
    if (Bits < 2)
      continue; // Not supported by `UnsignedDivisionByConstantInfo::get()`.
    if (Bits > 10)
      continue; // Unreasonably slow.
    EnumerateAPInts(Bits, [Bits](const APInt &Divisor) {
      if (Divisor.isZero())
        return; // Division by zero is undefined behavior.
      if (Divisor.isOne())
        return; // Division by one is the numerator.

      const UnsignedDivisionByConstantInfo Magics =
          UnsignedDivisionByConstantInfo::get(Divisor);
      EnumerateAPInts(Bits, [Divisor, Magics, Bits](const APInt &Numerator) {
        APInt NativeResult = Numerator.udiv(Divisor);
        for (bool LZOptimization : {true, false}) {
          for (bool AllowEvenDivisorOptimization : {true, false}) {
            for (bool ForceNPQ : {false, true}) {
              APInt MagicResult = UnsignedDivideUsingMagic(
                  Numerator, Divisor, LZOptimization,
                  AllowEvenDivisorOptimization, ForceNPQ, Magics);
              ASSERT_EQ(MagicResult, NativeResult)
                    << " ... given the operation:  urem i" << Bits << " "
                    << Numerator << ", " << Divisor
                    << " (allow LZ optimization = "
                    << LZOptimization << ", allow even divisior optimization = "
                    << AllowEvenDivisorOptimization << ", force NPQ = "
                    << ForceNPQ << ")";
            }
          }
        }
      });
    });
  }
}

TEST(UnsignedDivisionByConstantTest, WideningKinds) {
  {
    APInt Divisor(8, 7);
    auto Magics = UnsignedDivisionByConstantInfo::get(
        Divisor, /*LeadingZeros=*/0, /*AllowEvenDivisorOptimization=*/true,
        IntegerBitWidth::I16);
    EXPECT_EQ(Magics.Widening, UnsignedDivisionByConstantWidening::MulHigh);
    EXPECT_EQ(Magics.Magic.getBitWidth(), 16u);
    EXPECT_FALSE(Magics.IsAdd);
    EXPECT_EQ(Magics.PostShift, 0u);
  }

  {
    APInt Divisor(8, 7);
    auto Magics = UnsignedDivisionByConstantInfo::get(
        Divisor, /*LeadingZeros=*/0, /*AllowEvenDivisorOptimization=*/true,
        IntegerBitWidth::I64);
    EXPECT_EQ(Magics.Widening,
              UnsignedDivisionByConstantWidening::FullMultiply);
    EXPECT_EQ(Magics.Magic.getBitWidth(), 64u);
    EXPECT_FALSE(Magics.IsAdd);
    EXPECT_GT(Magics.PostShift, 0u);
  }

  {
    APInt Divisor(32, 7);
    auto Magics = UnsignedDivisionByConstantInfo::get(
        Divisor, /*LeadingZeros=*/0, /*AllowEvenDivisorOptimization=*/true,
        IntegerBitWidth::I64);
    EXPECT_EQ(Magics.Widening, UnsignedDivisionByConstantWidening::MulHigh);
    EXPECT_EQ(Magics.Magic.getBitWidth(), 64u);
    EXPECT_FALSE(Magics.IsAdd);
    EXPECT_EQ(Magics.PostShift, 0u);
  }
}

TEST(UnsignedDivisionByConstantTest, WidenedMagicExecutesCorrectly) {
  auto CheckAllNumerators = [](const APInt &Divisor,
                               IntegerBitWidth MaxBitWidth,
                               UnsignedDivisionByConstantWidening Widening) {
    auto Magics = UnsignedDivisionByConstantInfo::get(
        Divisor, /*LeadingZeros=*/0, /*AllowEvenDivisorOptimization=*/true,
        MaxBitWidth);
    ASSERT_EQ(Magics.Widening, Widening);
    EnumerateAPInts(Divisor.getBitWidth(), [&](const APInt &Numerator) {
      ASSERT_EQ(UnsignedDivideUsingMagic(Numerator, Divisor,
                                         /*LZOptimization=*/false,
                                         /*AllowEvenDivisorOptimization=*/true,
                                         /*ForceNPQ=*/false, Magics),
                Numerator.udiv(Divisor))
          << " ... given the operation: udiv i" << Divisor.getBitWidth() << " "
          << Numerator << ", " << Divisor << " with widening "
          << static_cast<int>(Widening);
    });
  };

  CheckAllNumerators(APInt(8, 7), IntegerBitWidth::I16,
                     UnsignedDivisionByConstantWidening::MulHigh);
  CheckAllNumerators(APInt(8, 7), IntegerBitWidth::I64,
                     UnsignedDivisionByConstantWidening::FullMultiply);
  CheckAllNumerators(APInt(16, 7), IntegerBitWidth::I64,
                     UnsignedDivisionByConstantWidening::FullMultiply);
}

} // end anonymous namespace
