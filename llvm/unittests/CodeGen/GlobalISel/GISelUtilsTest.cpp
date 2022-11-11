//===- GISelUtilsTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
static const LLT S1 = LLT::scalar(1);
static const LLT S8 = LLT::scalar(8);
static const LLT S16 = LLT::scalar(16);
static const LLT S32 = LLT::scalar(32);
static const LLT S64 = LLT::scalar(64);
static const LLT P0 = LLT::pointer(0, 64);
static const LLT P1 = LLT::pointer(1, 32);

static const LLT V2S8 = LLT::fixed_vector(2, 8);
static const LLT V4S8 = LLT::fixed_vector(4, 8);
static const LLT V8S8 = LLT::fixed_vector(8, 8);

static const LLT V2S16 = LLT::fixed_vector(2, 16);
static const LLT V3S16 = LLT::fixed_vector(3, 16);
static const LLT V4S16 = LLT::fixed_vector(4, 16);

static const LLT V2S32 = LLT::fixed_vector(2, 32);
static const LLT V3S32 = LLT::fixed_vector(3, 32);
static const LLT V4S32 = LLT::fixed_vector(4, 32);
static const LLT V6S32 = LLT::fixed_vector(6, 32);

static const LLT V2S64 = LLT::fixed_vector(2, 64);
static const LLT V3S64 = LLT::fixed_vector(3, 64);
static const LLT V4S64 = LLT::fixed_vector(4, 64);

static const LLT V2P0 = LLT::fixed_vector(2, P0);
static const LLT V3P0 = LLT::fixed_vector(3, P0);
static const LLT V4P0 = LLT::fixed_vector(4, P0);
static const LLT V6P0 = LLT::fixed_vector(6, P0);

static const LLT V2P1 = LLT::fixed_vector(2, P1);
static const LLT V4P1 = LLT::fixed_vector(4, P1);

TEST(GISelUtilsTest, getGCDType) {
  EXPECT_EQ(S1, getGCDType(S1, S1));
  EXPECT_EQ(S32, getGCDType(S32, S32));
  EXPECT_EQ(S1, getGCDType(S1, S32));
  EXPECT_EQ(S1, getGCDType(S32, S1));
  EXPECT_EQ(S16, getGCDType(S16, S32));
  EXPECT_EQ(S16, getGCDType(S32, S16));

  EXPECT_EQ(V2S32, getGCDType(V2S32, V2S32));
  EXPECT_EQ(S32, getGCDType(V3S32, V2S32));
  EXPECT_EQ(S32, getGCDType(V2S32, V3S32));

  EXPECT_EQ(V2S16, getGCDType(V4S16, V2S16));
  EXPECT_EQ(V2S16, getGCDType(V2S16, V4S16));

  EXPECT_EQ(V2S32, getGCDType(V4S32, V2S32));
  EXPECT_EQ(V2S32, getGCDType(V2S32, V4S32));

  EXPECT_EQ(S16, getGCDType(P0, S16));
  EXPECT_EQ(S16, getGCDType(S16, P0));

  EXPECT_EQ(S32, getGCDType(P0, S32));
  EXPECT_EQ(S32, getGCDType(S32, P0));

  EXPECT_EQ(P0, getGCDType(P0, S64));
  EXPECT_EQ(S64, getGCDType(S64, P0));

  EXPECT_EQ(S32, getGCDType(P0, P1));
  EXPECT_EQ(S32, getGCDType(P1, P0));

  EXPECT_EQ(P0, getGCDType(V3P0, V2P0));
  EXPECT_EQ(P0, getGCDType(V2P0, V3P0));

  EXPECT_EQ(P0, getGCDType(P0, V2P0));
  EXPECT_EQ(P0, getGCDType(V2P0, P0));


  EXPECT_EQ(V2P0, getGCDType(V2P0, V2P0));
  EXPECT_EQ(P0, getGCDType(V3P0, V2P0));
  EXPECT_EQ(P0, getGCDType(V2P0, V3P0));
  EXPECT_EQ(V2P0, getGCDType(V4P0, V2P0));

  EXPECT_EQ(V2P0, getGCDType(V2P0, V4P1));
  EXPECT_EQ(V4P1, getGCDType(V4P1, V2P0));

  EXPECT_EQ(V2P0, getGCDType(V4P0, V4P1));
  EXPECT_EQ(V4P1, getGCDType(V4P1, V4P0));

  // Elements have same size, but have different pointeriness, so prefer the
  // original element type.
  EXPECT_EQ(V2P0, getGCDType(V2P0, V4S64));
  EXPECT_EQ(V2S64, getGCDType(V4S64, V2P0));

  EXPECT_EQ(V2S16, getGCDType(V2S16, V4P1));
  EXPECT_EQ(P1, getGCDType(V4P1, V2S16));
  EXPECT_EQ(V2P1, getGCDType(V4P1, V4S16));
  EXPECT_EQ(V4S16, getGCDType(V4S16, V2P1));

  EXPECT_EQ(P0, getGCDType(P0, V2S64));
  EXPECT_EQ(S64, getGCDType(V2S64, P0));

  EXPECT_EQ(S16, getGCDType(V2S16, V3S16));
  EXPECT_EQ(S16, getGCDType(V3S16, V2S16));
  EXPECT_EQ(S16, getGCDType(V3S16, S16));
  EXPECT_EQ(S16, getGCDType(S16, V3S16));

  EXPECT_EQ(V2S16, getGCDType(V2S16, V2S32));
  EXPECT_EQ(S32, getGCDType(V2S32, V2S16));

  EXPECT_EQ(V4S8, getGCDType(V4S8, V2S32));
  EXPECT_EQ(S32, getGCDType(V2S32, V4S8));

  // Test cases where neither element type nicely divides.
  EXPECT_EQ(LLT::scalar(3),
            getGCDType(LLT::fixed_vector(3, 5), LLT::fixed_vector(2, 6)));
  EXPECT_EQ(LLT::scalar(3),
            getGCDType(LLT::fixed_vector(2, 6), LLT::fixed_vector(3, 5)));

  // Have to go smaller than a pointer element.
  EXPECT_EQ(LLT::scalar(3), getGCDType(LLT::fixed_vector(2, LLT::pointer(3, 6)),
                                       LLT::fixed_vector(3, 5)));
  EXPECT_EQ(LLT::scalar(3),
            getGCDType(LLT::fixed_vector(3, 5),
                       LLT::fixed_vector(2, LLT::pointer(3, 6))));

  EXPECT_EQ(V4S8, getGCDType(V4S8, S32));
  EXPECT_EQ(S32, getGCDType(S32, V4S8));
  EXPECT_EQ(V4S8, getGCDType(V4S8, P1));
  EXPECT_EQ(P1, getGCDType(P1, V4S8));

  EXPECT_EQ(V2S8, getGCDType(V2S8, V4S16));
  EXPECT_EQ(S16, getGCDType(V4S16, V2S8));

  EXPECT_EQ(S8, getGCDType(V2S8, LLT::fixed_vector(4, 2)));
  EXPECT_EQ(LLT::fixed_vector(4, 2), getGCDType(LLT::fixed_vector(4, 2), S8));

  EXPECT_EQ(LLT::pointer(4, 8),
            getGCDType(LLT::fixed_vector(2, LLT::pointer(4, 8)),
                       LLT::fixed_vector(4, 2)));

  EXPECT_EQ(LLT::fixed_vector(4, 2),
            getGCDType(LLT::fixed_vector(4, 2),
                       LLT::fixed_vector(2, LLT::pointer(4, 8))));

  EXPECT_EQ(LLT::scalar(4), getGCDType(LLT::fixed_vector(3, 4), S8));
  EXPECT_EQ(LLT::scalar(4), getGCDType(S8, LLT::fixed_vector(3, 4)));
}

TEST(GISelUtilsTest, getLCMType) {
  EXPECT_EQ(S1, getLCMType(S1, S1));
  EXPECT_EQ(S32, getLCMType(S32, S1));
  EXPECT_EQ(S32, getLCMType(S1, S32));
  EXPECT_EQ(S32, getLCMType(S32, S32));

  EXPECT_EQ(S32, getLCMType(S32, S16));
  EXPECT_EQ(S32, getLCMType(S16, S32));

  EXPECT_EQ(S64, getLCMType(S64, P0));
  EXPECT_EQ(P0, getLCMType(P0, S64));

  EXPECT_EQ(P0, getLCMType(S32, P0));
  EXPECT_EQ(P0, getLCMType(P0, S32));

  EXPECT_EQ(S32, getLCMType(S32, P1));
  EXPECT_EQ(P1, getLCMType(P1, S32));
  EXPECT_EQ(P0, getLCMType(P0, P0));
  EXPECT_EQ(P1, getLCMType(P1, P1));

  EXPECT_EQ(P0, getLCMType(P0, P1));
  EXPECT_EQ(P0, getLCMType(P1, P0));

  EXPECT_EQ(V2S32, getLCMType(V2S32, V2S32));
  EXPECT_EQ(V2S32, getLCMType(V2S32, S32));
  EXPECT_EQ(V2S32, getLCMType(S32, V2S32));
  EXPECT_EQ(V2S32, getLCMType(V2S32, V2S32));
  EXPECT_EQ(V6S32, getLCMType(V2S32, V3S32));
  EXPECT_EQ(V6S32, getLCMType(V3S32, V2S32));
  EXPECT_EQ(LLT::fixed_vector(12, S32), getLCMType(V4S32, V3S32));
  EXPECT_EQ(LLT::fixed_vector(12, S32), getLCMType(V3S32, V4S32));

  EXPECT_EQ(V2P0, getLCMType(V2P0, V2P0));
  EXPECT_EQ(V2P0, getLCMType(V2P0, P0));
  EXPECT_EQ(V2P0, getLCMType(P0, V2P0));
  EXPECT_EQ(V2P0, getLCMType(V2P0, V2P0));
  EXPECT_EQ(V6P0, getLCMType(V2P0, V3P0));
  EXPECT_EQ(V6P0, getLCMType(V3P0, V2P0));
  EXPECT_EQ(LLT::fixed_vector(12, P0), getLCMType(V4P0, V3P0));
  EXPECT_EQ(LLT::fixed_vector(12, P0), getLCMType(V3P0, V4P0));

  EXPECT_EQ(LLT::fixed_vector(12, S64), getLCMType(V4S64, V3P0));
  EXPECT_EQ(LLT::fixed_vector(12, P0), getLCMType(V3P0, V4S64));

  EXPECT_EQ(LLT::fixed_vector(12, P0), getLCMType(V4P0, V3S64));
  EXPECT_EQ(LLT::fixed_vector(12, S64), getLCMType(V3S64, V4P0));

  EXPECT_EQ(V2P0, getLCMType(V2P0, S32));
  EXPECT_EQ(V4S32, getLCMType(S32, V2P0));
  EXPECT_EQ(V2P0, getLCMType(V2P0, S64));
  EXPECT_EQ(V2S64, getLCMType(S64, V2P0));


  EXPECT_EQ(V2P0, getLCMType(V2P0, V2P1));
  EXPECT_EQ(V4P1, getLCMType(V2P1, V2P0));

  EXPECT_EQ(V2P0, getLCMType(V2P0, V4P1));
  EXPECT_EQ(V4P1, getLCMType(V4P1, V2P0));


  EXPECT_EQ(V2S32, getLCMType(V2S32, S64));
  EXPECT_EQ(S64, getLCMType(S64, V2S32));

  EXPECT_EQ(V4S16, getLCMType(V4S16, V2S32));
  EXPECT_EQ(V2S32, getLCMType(V2S32, V4S16));

  EXPECT_EQ(V2S32, getLCMType(V2S32, V4S8));
  EXPECT_EQ(V8S8, getLCMType(V4S8, V2S32));

  EXPECT_EQ(V2S16, getLCMType(V2S16, V4S8));
  EXPECT_EQ(V4S8, getLCMType(V4S8, V2S16));

  EXPECT_EQ(LLT::fixed_vector(6, S16), getLCMType(V3S16, V4S8));
  EXPECT_EQ(LLT::fixed_vector(12, S8), getLCMType(V4S8, V3S16));
  EXPECT_EQ(V4S16, getLCMType(V4S16, V4S8));
  EXPECT_EQ(V8S8, getLCMType(V4S8, V4S16));

  EXPECT_EQ(LLT::fixed_vector(6, 4), getLCMType(LLT::fixed_vector(3, 4), S8));
  EXPECT_EQ(LLT::fixed_vector(3, 8), getLCMType(S8, LLT::fixed_vector(3, 4)));

  EXPECT_EQ(LLT::fixed_vector(6, 4),
            getLCMType(LLT::fixed_vector(3, 4), LLT::pointer(4, 8)));
  EXPECT_EQ(LLT::fixed_vector(3, LLT::pointer(4, 8)),
            getLCMType(LLT::pointer(4, 8), LLT::fixed_vector(3, 4)));

  EXPECT_EQ(V2S64, getLCMType(V2S64, P0));
  EXPECT_EQ(V2P0, getLCMType(P0, V2S64));

  EXPECT_EQ(V2S64, getLCMType(V2S64, P1));
  EXPECT_EQ(V4P1, getLCMType(P1, V2S64));
}

TEST_F(AArch64GISelMITest, ConstFalseTest) {
  setUp();
  if (!TM)
    return;
  const auto &TLI = *B.getMF().getSubtarget().getTargetLowering();
  bool BooleanChoices[2] = {true, false};

  // AArch64 uses ZeroOrOneBooleanContent for scalars, and
  // ZeroOrNegativeOneBooleanContent for vectors.
  for (auto IsVec : BooleanChoices) {
    for (auto IsFP : BooleanChoices) {
      EXPECT_TRUE(isConstFalseVal(TLI, 0, IsVec, IsFP));
      EXPECT_FALSE(isConstFalseVal(TLI, 1, IsVec, IsFP));

      // This would be true with UndefinedBooleanContent.
      EXPECT_FALSE(isConstFalseVal(TLI, 2, IsVec, IsFP));
    }
  }
}
}
