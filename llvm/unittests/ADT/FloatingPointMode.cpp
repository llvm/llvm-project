//===- llvm/unittest/ADT/FloatingPointMode.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/FloatingPointMode.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(FloatingPointModeTest, ParseDenormalFPAttribute) {
  EXPECT_EQ(DenormalMode::IEEE, parseDenormalFPAttribute("ieee"));
  EXPECT_EQ(DenormalMode::IEEE, parseDenormalFPAttribute(""));
  EXPECT_EQ(DenormalMode::PreserveSign,
            parseDenormalFPAttribute("preserve-sign"));
  EXPECT_EQ(DenormalMode::PositiveZero,
            parseDenormalFPAttribute("positive-zero"));
  EXPECT_EQ(DenormalMode::Invalid, parseDenormalFPAttribute("foo"));
}

TEST(FloatingPointModeTest, DenormalAttributeName) {
  EXPECT_EQ("ieee", denormalModeName(DenormalMode::IEEE));
  EXPECT_EQ("preserve-sign", denormalModeName(DenormalMode::PreserveSign));
  EXPECT_EQ("positive-zero", denormalModeName(DenormalMode::PositiveZero));
  EXPECT_EQ("", denormalModeName(DenormalMode::Invalid));
}

}
