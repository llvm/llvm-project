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
#include <array>
#include <optional>

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
  APInt ShiftMask(Bits, -1);
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

APInt UnsignedDivideUsingMagic(APInt Numerator, APInt Divisor,
                               bool AllowEvenDivisorOptimization, bool ForceNPQ,
                               UnsignedDivisionByConstantInfo Magics) {
  unsigned Bits = Numerator.getBitWidth();

  unsigned PreShift = 0;
  if (AllowEvenDivisorOptimization) {
    // If the divisor is even, we can avoid using the expensive fixup by
    // shifting the divided value upfront.
    if (Magics.IsAdd && !Divisor[0]) {
      PreShift = Divisor.countTrailingZeros();
      // Get magic number for the shifted divisor.
      Magics =
          UnsignedDivisionByConstantInfo::get(Divisor.lshr(PreShift), PreShift);
      assert(!Magics.IsAdd && "Should use cheap fixup now");
    }
  }

  unsigned PostShift = 0;
  bool UseNPQ = false;
  if (!Magics.IsAdd || Divisor.isOne()) {
    assert(Magics.ShiftAmount < Divisor.getBitWidth() &&
           "We shouldn't generate an undefined shift!");
    PostShift = Magics.ShiftAmount;
    UseNPQ = false;
  } else {
    PostShift = Magics.ShiftAmount - 1;
    assert(PostShift < Divisor.getBitWidth() &&
           "We shouldn't generate an undefined shift!");
    UseNPQ = true;
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

  return Divisor.isOne() ? Numerator : Q;
}

TEST(UnsignedDivisionByConstantTest, Test) {
  for (unsigned Bits = 1; Bits <= 32; ++Bits) {
    if (Bits < 2)
      continue; // Not supported by `UnsignedDivisionByConstantInfo::get()`.
    if (Bits > 11)
      continue; // Unreasonably slow.
    EnumerateAPInts(Bits, [Bits](const APInt &Divisor) {
      if (Divisor.isZero())
        return; // Division by zero is undefined behavior.
      const UnsignedDivisionByConstantInfo Magics =
          UnsignedDivisionByConstantInfo::get(Divisor);
      EnumerateAPInts(Bits, [Divisor, Magics, Bits](const APInt &Numerator) {
        APInt NativeResult = Numerator.udiv(Divisor);
        for (bool AllowEvenDivisorOptimization : {true, false}) {
          for (bool ForceNPQ : {false, true}) {
            APInt MagicResult = UnsignedDivideUsingMagic(
                Numerator, Divisor, AllowEvenDivisorOptimization, ForceNPQ,
                Magics);
            ASSERT_EQ(MagicResult, NativeResult)
                << " ... given the operation:  urem i" << Bits << " "
                << Numerator << ", " << Divisor
                << " (allow even divisior optimization = "
                << AllowEvenDivisorOptimization << ", force NPQ = " << ForceNPQ
                << ")";
          }
        }
      });
    });
  }
}

} // end anonymous namespace
