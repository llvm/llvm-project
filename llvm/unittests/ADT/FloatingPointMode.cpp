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

TEST(FloatingPointModeTest, ParseDenormalFPAttributeComponent) {
  EXPECT_EQ(DenormalMode::IEEE, parseDenormalFPAttributeComponent("ieee"));
  EXPECT_EQ(DenormalMode::IEEE, parseDenormalFPAttributeComponent(""));
  EXPECT_EQ(DenormalMode::PreserveSign,
            parseDenormalFPAttributeComponent("preserve-sign"));
  EXPECT_EQ(DenormalMode::PositiveZero,
            parseDenormalFPAttributeComponent("positive-zero"));
  EXPECT_EQ(DenormalMode::Invalid, parseDenormalFPAttributeComponent("foo"));
}

TEST(FloatingPointModeTest, DenormalAttributeName) {
  EXPECT_EQ("ieee", denormalModeKindName(DenormalMode::IEEE));
  EXPECT_EQ("preserve-sign", denormalModeKindName(DenormalMode::PreserveSign));
  EXPECT_EQ("positive-zero", denormalModeKindName(DenormalMode::PositiveZero));
  EXPECT_EQ("", denormalModeKindName(DenormalMode::Invalid));
}

TEST(FloatingPointModeTest, ParseDenormalFPAttribute) {
  EXPECT_EQ(DenormalMode(DenormalMode::IEEE, DenormalMode::IEEE),
            parseDenormalFPAttribute("ieee"));
  EXPECT_EQ(DenormalMode(DenormalMode::IEEE, DenormalMode::IEEE),
            parseDenormalFPAttribute("ieee,ieee"));
  EXPECT_EQ(DenormalMode(DenormalMode::IEEE, DenormalMode::IEEE),
            parseDenormalFPAttribute("ieee,"));
  EXPECT_EQ(DenormalMode(DenormalMode::IEEE, DenormalMode::IEEE),
            parseDenormalFPAttribute(""));
  EXPECT_EQ(DenormalMode(DenormalMode::IEEE, DenormalMode::IEEE),
            parseDenormalFPAttribute(","));

  EXPECT_EQ(DenormalMode(DenormalMode::PreserveSign, DenormalMode::PreserveSign),
            parseDenormalFPAttribute("preserve-sign"));
  EXPECT_EQ(DenormalMode(DenormalMode::PreserveSign, DenormalMode::PreserveSign),
            parseDenormalFPAttribute("preserve-sign,"));
  EXPECT_EQ(DenormalMode(DenormalMode::PreserveSign, DenormalMode::PreserveSign),
            parseDenormalFPAttribute("preserve-sign,preserve-sign"));

  EXPECT_EQ(DenormalMode(DenormalMode::PositiveZero, DenormalMode::PositiveZero),
            parseDenormalFPAttribute("positive-zero"));
  EXPECT_EQ(DenormalMode(DenormalMode::PositiveZero, DenormalMode::PositiveZero),
            parseDenormalFPAttribute("positive-zero,positive-zero"));


  EXPECT_EQ(DenormalMode(DenormalMode::IEEE, DenormalMode::PositiveZero),
            parseDenormalFPAttribute("ieee,positive-zero"));
  EXPECT_EQ(DenormalMode(DenormalMode::PositiveZero, DenormalMode::IEEE),
            parseDenormalFPAttribute("positive-zero,ieee"));

  EXPECT_EQ(DenormalMode(DenormalMode::PreserveSign, DenormalMode::IEEE),
            parseDenormalFPAttribute("preserve-sign,ieee"));
  EXPECT_EQ(DenormalMode(DenormalMode::IEEE, DenormalMode::PreserveSign),
            parseDenormalFPAttribute("ieee,preserve-sign"));


  EXPECT_EQ(DenormalMode(DenormalMode::Invalid, DenormalMode::Invalid),
            parseDenormalFPAttribute("foo"));
  EXPECT_EQ(DenormalMode(DenormalMode::Invalid, DenormalMode::Invalid),
            parseDenormalFPAttribute("foo,foo"));
  EXPECT_EQ(DenormalMode(DenormalMode::Invalid, DenormalMode::Invalid),
            parseDenormalFPAttribute("foo,bar"));
}

TEST(FloatingPointModeTest, RenderDenormalFPAttribute) {
  EXPECT_EQ(DenormalMode(DenormalMode::Invalid, DenormalMode::Invalid),
            parseDenormalFPAttribute("foo"));

  EXPECT_EQ("ieee,ieee",
            DenormalMode(DenormalMode::IEEE, DenormalMode::IEEE).str());
  EXPECT_EQ(",",
            DenormalMode(DenormalMode::Invalid, DenormalMode::Invalid).str());

  EXPECT_EQ(
    "preserve-sign,preserve-sign",
    DenormalMode(DenormalMode::PreserveSign, DenormalMode::PreserveSign).str());

  EXPECT_EQ(
    "positive-zero,positive-zero",
    DenormalMode(DenormalMode::PositiveZero, DenormalMode::PositiveZero).str());

  EXPECT_EQ(
    "ieee,preserve-sign",
    DenormalMode(DenormalMode::IEEE, DenormalMode::PreserveSign).str());

  EXPECT_EQ(
    "preserve-sign,ieee",
    DenormalMode(DenormalMode::PreserveSign, DenormalMode::IEEE).str());

  EXPECT_EQ(
    "preserve-sign,positive-zero",
    DenormalMode(DenormalMode::PreserveSign, DenormalMode::PositiveZero).str());
}

TEST(FloatingPointModeTest, DenormalModeIsSimple) {
  EXPECT_TRUE(DenormalMode(DenormalMode::IEEE, DenormalMode::IEEE).isSimple());
  EXPECT_FALSE(DenormalMode(DenormalMode::IEEE,
                            DenormalMode::Invalid).isSimple());
  EXPECT_FALSE(DenormalMode(DenormalMode::PreserveSign,
                            DenormalMode::PositiveZero).isSimple());
}

TEST(FloatingPointModeTest, DenormalModeIsValid) {
  EXPECT_TRUE(DenormalMode(DenormalMode::IEEE, DenormalMode::IEEE).isValid());
  EXPECT_FALSE(DenormalMode(DenormalMode::IEEE, DenormalMode::Invalid).isValid());
  EXPECT_FALSE(DenormalMode(DenormalMode::Invalid, DenormalMode::IEEE).isValid());
  EXPECT_FALSE(DenormalMode(DenormalMode::Invalid,
                            DenormalMode::Invalid).isValid());
}

TEST(FloatingPointModeTest, DenormalModeConstructor) {
  EXPECT_EQ(DenormalMode(DenormalMode::Invalid, DenormalMode::Invalid),
            DenormalMode::getInvalid());
  EXPECT_EQ(DenormalMode(DenormalMode::IEEE, DenormalMode::IEEE),
            DenormalMode::getIEEE());
  EXPECT_EQ(DenormalMode(DenormalMode::PreserveSign, DenormalMode::PreserveSign),
            DenormalMode::getPreserveSign());
  EXPECT_EQ(DenormalMode(DenormalMode::PositiveZero, DenormalMode::PositiveZero),
            DenormalMode::getPositiveZero());
}

}
