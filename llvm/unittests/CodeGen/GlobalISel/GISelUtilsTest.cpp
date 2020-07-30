//===- GISelUtilsTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
static const LLT S1 = LLT::scalar(1);
static const LLT S16 = LLT::scalar(16);
static const LLT S32 = LLT::scalar(32);
static const LLT S64 = LLT::scalar(64);
static const LLT P0 = LLT::pointer(0, 64);
static const LLT P1 = LLT::pointer(1, 32);

static const LLT V2S16 = LLT::vector(2, 16);
static const LLT V4S16 = LLT::vector(4, 16);

static const LLT V2S32 = LLT::vector(2, 32);
static const LLT V3S32 = LLT::vector(3, 32);
static const LLT V4S32 = LLT::vector(4, 32);
static const LLT V6S32 = LLT::vector(6, 32);

static const LLT V2P0 = LLT::vector(2, P0);
static const LLT V3P0 = LLT::vector(3, P0);
static const LLT V4P0 = LLT::vector(4, P0);
static const LLT V6P0 = LLT::vector(6, P0);

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

  EXPECT_EQ(S64, getGCDType(P0, S64));
  EXPECT_EQ(S64, getGCDType(S64, P0));

  EXPECT_EQ(S32, getGCDType(P0, P1));
  EXPECT_EQ(S32, getGCDType(P1, P0));

  EXPECT_EQ(P0, getGCDType(V3P0, V2P0));
  EXPECT_EQ(P0, getGCDType(V2P0, V3P0));
}

TEST(GISelUtilsTest, getLCMType) {
  EXPECT_EQ(S1, getLCMType(S1, S1));
  EXPECT_EQ(S32, getLCMType(S32, S1));
  EXPECT_EQ(S32, getLCMType(S1, S32));
  EXPECT_EQ(S32, getLCMType(S32, S32));

  EXPECT_EQ(S32, getLCMType(S32, S16));
  EXPECT_EQ(S32, getLCMType(S16, S32));

  EXPECT_EQ(S64, getLCMType(S64, P0));
  EXPECT_EQ(S64, getLCMType(P0, S64));

  EXPECT_EQ(S64, getLCMType(S32, P0));
  EXPECT_EQ(S64, getLCMType(P0, S32));

  EXPECT_EQ(S32, getLCMType(S32, P1));
  EXPECT_EQ(S32, getLCMType(P1, S32));
  EXPECT_EQ(S64, getLCMType(P0, P0));
  EXPECT_EQ(S32, getLCMType(P1, P1));

  EXPECT_EQ(S64, getLCMType(P0, P1));
  EXPECT_EQ(S64, getLCMType(P1, P0));

  EXPECT_EQ(V2S32, getLCMType(V2S32, V2S32));
  EXPECT_EQ(V2S32, getLCMType(V2S32, S32));
  EXPECT_EQ(V2S32, getLCMType(S32, V2S32));
  EXPECT_EQ(V2S32, getLCMType(V2S32, V2S32));
  EXPECT_EQ(V6S32, getLCMType(V2S32, V3S32));
  EXPECT_EQ(V6S32, getLCMType(V3S32, V2S32));
  EXPECT_EQ(LLT::vector(12, S32), getLCMType(V4S32, V3S32));
  EXPECT_EQ(LLT::vector(12, S32), getLCMType(V3S32, V4S32));

  EXPECT_EQ(V2P0, getLCMType(V2P0, V2P0));
  EXPECT_EQ(V2P0, getLCMType(V2P0, P0));
  EXPECT_EQ(V2P0, getLCMType(P0, V2P0));
  EXPECT_EQ(V2P0, getLCMType(V2P0, V2P0));
  EXPECT_EQ(V6P0, getLCMType(V2P0, V3P0));
  EXPECT_EQ(V6P0, getLCMType(V3P0, V2P0));
  EXPECT_EQ(LLT::vector(12, P0), getLCMType(V4P0, V3P0));
  EXPECT_EQ(LLT::vector(12, P0), getLCMType(V3P0, V4P0));

  // FIXME
  // EXPECT_EQ(V2S32, getLCMType(V2S32, S64));

  // FIXME
  //EXPECT_EQ(S64, getLCMType(S64, V2S32));
}

}
