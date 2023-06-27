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
  EXPECT_EQ(DenormalMode::Dynamic,
            parseDenormalFPAttributeComponent("dynamic"));
  EXPECT_EQ(DenormalMode::Invalid, parseDenormalFPAttributeComponent("foo"));
}

TEST(FloatingPointModeTest, DenormalAttributeName) {
  EXPECT_EQ("ieee", denormalModeKindName(DenormalMode::IEEE));
  EXPECT_EQ("preserve-sign", denormalModeKindName(DenormalMode::PreserveSign));
  EXPECT_EQ("positive-zero", denormalModeKindName(DenormalMode::PositiveZero));
  EXPECT_EQ("dynamic", denormalModeKindName(DenormalMode::Dynamic));
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

  EXPECT_EQ(DenormalMode(DenormalMode::Dynamic, DenormalMode::Dynamic),
            parseDenormalFPAttribute("dynamic"));
  EXPECT_EQ(DenormalMode(DenormalMode::Dynamic, DenormalMode::Dynamic),
            parseDenormalFPAttribute("dynamic,dynamic"));

  EXPECT_EQ(DenormalMode(DenormalMode::IEEE, DenormalMode::PositiveZero),
            parseDenormalFPAttribute("ieee,positive-zero"));
  EXPECT_EQ(DenormalMode(DenormalMode::PositiveZero, DenormalMode::IEEE),
            parseDenormalFPAttribute("positive-zero,ieee"));

  EXPECT_EQ(DenormalMode(DenormalMode::PreserveSign, DenormalMode::IEEE),
            parseDenormalFPAttribute("preserve-sign,ieee"));
  EXPECT_EQ(DenormalMode(DenormalMode::IEEE, DenormalMode::PreserveSign),
            parseDenormalFPAttribute("ieee,preserve-sign"));

  EXPECT_EQ(DenormalMode(DenormalMode::Dynamic, DenormalMode::PreserveSign),
            parseDenormalFPAttribute("dynamic,preserve-sign"));
  EXPECT_EQ(DenormalMode(DenormalMode::PreserveSign, DenormalMode::Dynamic),
            parseDenormalFPAttribute("preserve-sign,dynamic"));

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

  EXPECT_EQ("dynamic,dynamic",
            DenormalMode(DenormalMode::Dynamic, DenormalMode::Dynamic).str());
  EXPECT_EQ("ieee,dynamic",
            DenormalMode(DenormalMode::IEEE, DenormalMode::Dynamic).str());
  EXPECT_EQ("dynamic,ieee",
            DenormalMode(DenormalMode::Dynamic, DenormalMode::IEEE).str());
}

TEST(FloatingPointModeTest, DenormalModeIsSimple) {
  EXPECT_TRUE(DenormalMode(DenormalMode::IEEE, DenormalMode::IEEE).isSimple());
  EXPECT_FALSE(DenormalMode(DenormalMode::IEEE,
                            DenormalMode::Invalid).isSimple());
  EXPECT_FALSE(DenormalMode(DenormalMode::PreserveSign,
                            DenormalMode::PositiveZero).isSimple());
  EXPECT_FALSE(DenormalMode(DenormalMode::PreserveSign, DenormalMode::Dynamic)
                   .isSimple());
  EXPECT_FALSE(DenormalMode(DenormalMode::Dynamic, DenormalMode::PreserveSign)
                   .isSimple());
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
  EXPECT_EQ(DenormalMode::getIEEE(), DenormalMode::getDefault());
  EXPECT_EQ(DenormalMode(DenormalMode::Dynamic, DenormalMode::Dynamic),
            DenormalMode::getDynamic());
  EXPECT_EQ(DenormalMode(DenormalMode::PreserveSign, DenormalMode::PreserveSign),
            DenormalMode::getPreserveSign());
  EXPECT_EQ(DenormalMode(DenormalMode::PositiveZero, DenormalMode::PositiveZero),
            DenormalMode::getPositiveZero());
}

TEST(FloatingPointModeTest, DenormalModeMerge) {
  EXPECT_EQ(
      DenormalMode::getInvalid(),
      DenormalMode::getInvalid().mergeCalleeMode(DenormalMode::getInvalid()));
  EXPECT_EQ(DenormalMode::getIEEE(), DenormalMode::getInvalid().mergeCalleeMode(
                                         DenormalMode::getIEEE()));
  EXPECT_EQ(DenormalMode::getInvalid(), DenormalMode::getIEEE().mergeCalleeMode(
                                            DenormalMode::getInvalid()));

  EXPECT_EQ(DenormalMode::getIEEE(), DenormalMode::getIEEE().mergeCalleeMode(
                                         DenormalMode::getDynamic()));
  EXPECT_EQ(DenormalMode::getPreserveSign(),
            DenormalMode::getPreserveSign().mergeCalleeMode(
                DenormalMode::getDynamic()));
  EXPECT_EQ(DenormalMode::getPositiveZero(),
            DenormalMode::getPositiveZero().mergeCalleeMode(
                DenormalMode::getDynamic()));
  EXPECT_EQ(
      DenormalMode::getDynamic(),
      DenormalMode::getDynamic().mergeCalleeMode(DenormalMode::getDynamic()));

  EXPECT_EQ(DenormalMode(DenormalMode::IEEE, DenormalMode::PreserveSign),
            DenormalMode(DenormalMode::IEEE, DenormalMode::PreserveSign)
                .mergeCalleeMode(
                    DenormalMode(DenormalMode::IEEE, DenormalMode::Dynamic)));

  EXPECT_EQ(DenormalMode(DenormalMode::PreserveSign, DenormalMode::IEEE),
            DenormalMode(DenormalMode::PreserveSign, DenormalMode::IEEE)
                .mergeCalleeMode(
                    DenormalMode(DenormalMode::Dynamic, DenormalMode::IEEE)));

  EXPECT_EQ(
      DenormalMode(DenormalMode::PositiveZero, DenormalMode::PreserveSign),
      DenormalMode(DenormalMode::PositiveZero, DenormalMode::PreserveSign)
          .mergeCalleeMode(
              DenormalMode(DenormalMode::Dynamic, DenormalMode::Dynamic)));

  EXPECT_EQ(
      DenormalMode(DenormalMode::PositiveZero, DenormalMode::PreserveSign),
      DenormalMode(DenormalMode::PositiveZero, DenormalMode::PreserveSign)
          .mergeCalleeMode(
              DenormalMode(DenormalMode::PositiveZero, DenormalMode::Dynamic)));

  EXPECT_EQ(
      DenormalMode(DenormalMode::PositiveZero, DenormalMode::PreserveSign),
      DenormalMode(DenormalMode::PositiveZero, DenormalMode::PreserveSign)
          .mergeCalleeMode(
              DenormalMode(DenormalMode::Dynamic, DenormalMode::PreserveSign)));

  // Test some invalid / undefined behavior cases
  EXPECT_EQ(
      DenormalMode::getPreserveSign(),
      DenormalMode::getIEEE().mergeCalleeMode(DenormalMode::getPreserveSign()));
  EXPECT_EQ(
      DenormalMode::getPreserveSign(),
      DenormalMode::getIEEE().mergeCalleeMode(DenormalMode::getPreserveSign()));
  EXPECT_EQ(
      DenormalMode::getIEEE(),
      DenormalMode::getPreserveSign().mergeCalleeMode(DenormalMode::getIEEE()));
  EXPECT_EQ(
      DenormalMode::getIEEE(),
      DenormalMode::getPreserveSign().mergeCalleeMode(DenormalMode::getIEEE()));
}

TEST(FloatingPointModeTest, DenormalModePredicates) {
  EXPECT_TRUE(DenormalMode::getPreserveSign().inputsAreZero());
  EXPECT_TRUE(DenormalMode::getPositiveZero().inputsAreZero());
  EXPECT_FALSE(DenormalMode::getIEEE().inputsAreZero());
  EXPECT_FALSE(DenormalMode::getDynamic().inputsAreZero());
}
}
