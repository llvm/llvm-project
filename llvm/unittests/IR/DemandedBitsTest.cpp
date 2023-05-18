//===- DemandedBitsTest.cpp - DemandedBits tests --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DemandedBits.h"
#include "../Support/KnownBitsTest.h"
#include "llvm/Support/KnownBits.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

template <typename Fn1, typename Fn2>
static void TestBinOpExhaustive(Fn1 PropagateFn, Fn2 EvalFn) {
  unsigned Bits = 4;
  unsigned Max = 1 << Bits;
  ForeachKnownBits(Bits, [&](const KnownBits &Known1) {
    ForeachKnownBits(Bits, [&](const KnownBits &Known2) {
      for (unsigned AOut_ = 0; AOut_ < Max; AOut_++) {
        APInt AOut(Bits, AOut_);
        APInt AB1 = PropagateFn(0, AOut, Known1, Known2);
        APInt AB2 = PropagateFn(1, AOut, Known1, Known2);
        {
          // If the propagator claims that certain known bits
          // didn't matter, check it doesn't change its mind
          // when they become unknown.
          KnownBits Known1Redacted;
          KnownBits Known2Redacted;
          Known1Redacted.Zero = Known1.Zero & AB1;
          Known1Redacted.One = Known1.One & AB1;
          Known2Redacted.Zero = Known2.Zero & AB2;
          Known2Redacted.One = Known2.One & AB2;

          APInt AB1R = PropagateFn(0, AOut, Known1Redacted, Known2Redacted);
          APInt AB2R = PropagateFn(1, AOut, Known1Redacted, Known2Redacted);
          EXPECT_EQ(AB1, AB1R);
          EXPECT_EQ(AB2, AB2R);
        }
        ForeachNumInKnownBits(Known1, [&](APInt Value1) {
          ForeachNumInKnownBits(Known2, [&](APInt Value2) {
            APInt ReferenceResult = EvalFn((Value1 & AB1), (Value2 & AB2));
            APInt Result = EvalFn(Value1, Value2);
            EXPECT_EQ(Result & AOut, ReferenceResult & AOut);
          });
        });
      }
    });
  });
}

TEST(DemandedBitsTest, Add) {
  TestBinOpExhaustive(DemandedBits::determineLiveOperandBitsAdd,
                      [](APInt N1, APInt N2) -> APInt { return N1 + N2; });
}

TEST(DemandedBitsTest, Sub) {
  TestBinOpExhaustive(DemandedBits::determineLiveOperandBitsSub,
                      [](APInt N1, APInt N2) -> APInt { return N1 - N2; });
}

} // anonymous namespace
