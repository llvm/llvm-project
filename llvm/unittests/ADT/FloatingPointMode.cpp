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

static constexpr FPClassTest OrderedLessThanZeroMask =
    fcNegSubnormal | fcNegNormal | fcNegInf;
static constexpr FPClassTest OrderedGreaterThanZeroMask =
    fcPosSubnormal | fcPosNormal | fcPosInf;

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

#define TEST_ORDERED_LT(a, b)                                                  \
  EXPECT_TRUE(cannotOrderStrictlyGreater(a, b));                               \
  EXPECT_FALSE(cannotOrderStrictlyGreater(b, a));

TEST(FloatingPointModeTest, cannotOrderStrictlyGreater) {
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNone, fcNone));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcAllFlags, fcAllFlags));

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNan, fcNan));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcQNan, fcQNan));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcSNan, fcSNan));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcSNan, fcQNan));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcQNan, fcSNan));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcQNan, fcNan));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcSNan, fcNan));

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegInf, fcSNan));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegInf, fcQNan));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegInf, fcNan));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcSNan, fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcQNan, fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNan, fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegInf, fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegZero, fcPosZero));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegZero, fcPosZero, true));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcPosZero, fcNegZero));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcPosZero, fcNegZero, true));

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcPosZero, fcPosZero));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcPosZero, fcPosZero, true));

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegZero, fcNegZero));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegZero, fcNegZero, true));

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegSubnormal, fcNegZero));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcNegZero, fcNegSubnormal));

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegZero, fcPosSubnormal));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcPosSubnormal, fcNegZero));

  EXPECT_FALSE(cannotOrderStrictlyGreater(fcNegZero, fcSubnormal));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcSubnormal, fcNegZero));

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcPosZero, fcPosSubnormal));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcPosSubnormal, fcPosZero));

  EXPECT_FALSE(cannotOrderStrictlyGreater(fcPosZero, fcNegSubnormal));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcPosZero, fcNegSubnormal, true));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcZero, fcSubnormal));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcZero, fcSubnormal, true));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcZero, fcNegSubnormal));

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcZero, fcPosSubnormal));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcPosSubnormal, fcZero));

  TEST_ORDERED_LT(fcNegInf, fcNegNormal);
  TEST_ORDERED_LT(fcNegInf, fcNegSubnormal);
  TEST_ORDERED_LT(fcNegInf, fcNegZero);
  TEST_ORDERED_LT(fcNegInf, fcPosZero);
  TEST_ORDERED_LT(fcNegInf, fcPosSubnormal);
  TEST_ORDERED_LT(fcNegInf, fcPosNormal);
  TEST_ORDERED_LT(fcNegInf, fcPosInf);

  TEST_ORDERED_LT(fcNegNormal, fcPosNormal);
  TEST_ORDERED_LT(fcNegNormal, fcPositive);
  TEST_ORDERED_LT(fcNegNormal, fcPositive | fcNan);
  TEST_ORDERED_LT(fcNegNormal | fcNan, fcPositive | fcNan);

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegNormal, fcPosNormal | fcNan));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcPosNormal | fcNan, fcNegNormal));
  EXPECT_FALSE(
      cannotOrderStrictlyGreater(fcPosNormal | fcNan, fcNegNormal | fcNan));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcNegNormal | fcNan, fcNegNormal));
  EXPECT_FALSE(
      cannotOrderStrictlyGreater(fcNegNormal | fcNan, fcNegNormal | fcNan));

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegInf, ~fcNegInf));
  EXPECT_FALSE(cannotOrderStrictlyGreater(~fcNegInf, fcNegInf));

  TEST_ORDERED_LT(fcNegInf, ~(fcNegInf | fcNan));
  TEST_ORDERED_LT(fcNegative, fcPositive);
  TEST_ORDERED_LT(fcNegFinite, fcPosFinite);
  TEST_ORDERED_LT(fcNegZero, fcPosInf);
  TEST_ORDERED_LT(fcPosZero, fcPosInf);
  TEST_ORDERED_LT(fcZero, fcPosInf);

  EXPECT_FALSE(cannotOrderStrictlyGreater(fcNegZero, fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegInf, fcNegZero));

  EXPECT_FALSE(cannotOrderStrictlyGreater(fcZero, fcInf));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcInf, fcZero));

  EXPECT_FALSE(cannotOrderStrictlyGreater(fcZero, fcInf | fcNan));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcInf | fcNan, fcZero));

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegInf | fcNan, fcZero));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcPosInf | fcNan, fcZero));

  EXPECT_FALSE(cannotOrderStrictlyGreater(fcZero | fcNan, fcInf));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcInf, fcZero | fcNan));

  TEST_ORDERED_LT(OrderedLessThanZeroMask, OrderedGreaterThanZeroMask);
  TEST_ORDERED_LT(OrderedLessThanZeroMask, OrderedGreaterThanZeroMask | fcNan);

  TEST_ORDERED_LT(OrderedLessThanZeroMask | fcNegZero,
                  OrderedGreaterThanZeroMask);
  TEST_ORDERED_LT(OrderedLessThanZeroMask | fcPosZero,
                  OrderedGreaterThanZeroMask);
  TEST_ORDERED_LT(OrderedLessThanZeroMask | fcZero, OrderedGreaterThanZeroMask);

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegZero, fcPosZero));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegZero, fcPosZero, true));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcPosZero, fcNegZero));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcPosZero, fcNegZero, true));

  TEST_ORDERED_LT(fcNegZero, fcPosSubnormal);
  TEST_ORDERED_LT(fcNegZero, fcPosZero | fcPosSubnormal);
  TEST_ORDERED_LT(fcNegZero, OrderedGreaterThanZeroMask);

  EXPECT_FALSE(cannotOrderStrictlyGreater(OrderedLessThanZeroMask,
                                          OrderedLessThanZeroMask | fcNan));
  EXPECT_FALSE(cannotOrderStrictlyGreater(OrderedLessThanZeroMask | fcNan,
                                          OrderedLessThanZeroMask | fcNan));

  TEST_ORDERED_LT(fcZero, fcPosInf);
  TEST_ORDERED_LT(fcZero, fcPosInf | fcNan);

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcZero | fcNan, fcPosInf));

  TEST_ORDERED_LT(fcPosNormal, fcPosInf);
  TEST_ORDERED_LT(fcPosNormal, fcPosInf | fcNan);

  TEST_ORDERED_LT(fcNegInf, fcPosInf);
  TEST_ORDERED_LT(fcNegInf | fcNegZero, fcPosInf);

  EXPECT_TRUE(
      cannotOrderStrictlyGreater(fcNegInf | fcNegZero, fcZero | fcPosInf));
  EXPECT_FALSE(
      cannotOrderStrictlyGreater(fcZero | fcPosInf, fcNegInf | fcNegZero));

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegInf | fcNegZero,
                                         fcZero | fcPosInf, true));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcZero | fcPosInf,
                                          fcNegInf | fcNegZero, true));

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegInf | fcNegZero,
                                         fcZero | fcPosInf | fcNan));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegInf | fcNegZero,
                                         fcPosZero | fcPosInf | fcNan));

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcPosInf, fcPosInf));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegInf, fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcPosZero, fcPosZero));
  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegZero, fcNegZero));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcNegSubnormal, fcNegSubnormal));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcPosSubnormal, fcPosSubnormal));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcNegNormal, fcNegNormal));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcPosNormal, fcPosNormal));

  EXPECT_FALSE(cannotOrderStrictlyGreater(fcFinite, fcFinite));

  EXPECT_FALSE(cannotOrderStrictlyGreater(fcFinite, fcInf));
  EXPECT_FALSE(cannotOrderStrictlyGreater(fcInf, fcInf));

  EXPECT_FALSE(cannotOrderStrictlyGreater(fcPosInf, ~fcPosInf));
  EXPECT_TRUE(cannotOrderStrictlyGreater(~fcPosInf, fcPosInf));

  EXPECT_TRUE(cannotOrderStrictlyGreater(fcNegInf, ~fcNegInf));
  EXPECT_FALSE(cannotOrderStrictlyGreater(~fcNegInf, fcNegInf));
}

TEST(FloatingPointModeTest, cannotOrderStrictlyGreaterEq) {
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNone, fcNone));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcAllFlags, fcAllFlags));

  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNan, fcNan));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcQNan, fcQNan));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcSNan, fcSNan));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcSNan, fcQNan));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcQNan, fcSNan));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcQNan, fcNan));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcSNan, fcNan));

  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNegInf, fcSNan));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNegInf, fcQNan));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNegInf, fcNan));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcSNan, fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcQNan, fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNan, fcNegInf));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegInf, fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNegZero, fcPosZero,
                                           /*OrderedZeroSign=*/true));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegZero, fcPosZero));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosZero, fcNegZero));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosZero, fcNegZero,
                                            /*OrderedZeroSign=*/true));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosZero, fcPosZero));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosZero, fcPosZero,
                                            /*OrderedZeroSign=*/true));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegZero, fcNegZero));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegZero, fcNegZero,
                                            /*OrderedZeroSign=*/true));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(OrderedLessThanZeroMask,
                                            OrderedLessThanZeroMask));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(OrderedLessThanZeroMask,
                                            OrderedLessThanZeroMask | fcNan));

  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(OrderedLessThanZeroMask,
                                           OrderedGreaterThanZeroMask));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(OrderedLessThanZeroMask,
                                           OrderedGreaterThanZeroMask | fcNan));

  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(OrderedLessThanZeroMask | fcNan,
                                           OrderedGreaterThanZeroMask));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegZero, fcNegZero | fcNan));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegZero | fcNan, fcNegZero));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegZero, fcPosZero));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNegZero, fcPosZero,
                                           /*OrderedZeroSign=*/true));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosZero, fcNegZero));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegZero | fcNan, fcPosZero));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNegZero | fcNan, fcPosZero,
                                           /*OrderedZeroSign=*/true));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosZero | fcNan, fcNegZero));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegZero, fcPosZero | fcNan));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNegZero, fcPosZero | fcNan,
                                           /*OrderedZeroSign=*/true));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosZero, fcNegZero | fcNan));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegZero, fcZero));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcZero, fcNegZero,
                                            /*OrderedZeroSign=*/true));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcZero, fcPosZero));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcZero, fcPosZero,
                                            /*OrderedZeroSign=*/true));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosInf, fcPosInf));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosInf, fcPosInf | fcNan));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosInf | fcNan, fcPosInf));
  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcPosInf | fcNan, fcPosInf | fcNan));

  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNegSubnormal, fcPosZero));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNegSubnormal, fcNegZero));
  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcNegSubnormal | fcNegZero, fcNegZero));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosSubnormal, fcZero));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNegSubnormal, fcZero));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcSubnormal, fcZero));

  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcZero, fcPosInf));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcZero, fcPosInf | fcNan));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcZero | fcNan, fcPosInf));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosNormal, fcPosNormal));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosSubnormal, fcPosSubnormal));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegNormal, fcNegNormal));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegSubnormal, fcNegSubnormal));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosNormal | fcPosSubnormal,
                                            fcPosNormal | fcPosSubnormal));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegNormal | fcNegSubnormal,
                                            fcNegNormal | fcNegSubnormal));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNormal, fcNormal));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcSubnormal, fcSubnormal));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcSubnormal | fcNormal,
                                            fcSubnormal | fcNormal));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegInf, fcNegInf | fcNan));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegInf | fcNan, fcNegInf));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegSubnormal, fcNegNormal));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosSubnormal, fcNegNormal));

  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcPosSubnormal, fcPosNormal));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNegSubnormal, fcPosNormal));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegSubnormal, fcNegInf));

  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcNegInf | fcNegZero, fcZero | fcPosInf));
  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcZero | fcPosInf, fcNegInf | fcNegZero));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegInf | fcNegZero,
                                            fcZero | fcPosInf | fcNan));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegInf | fcNegZero,
                                            fcPosZero | fcPosInf | fcNan));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNegInf | fcNegZero,
                                           fcPosZero | fcPosInf | fcNan,
                                           /*OrderedZeroSign=*/true));

  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcPosInf | fcPosNormal, fcPosNormal));
  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcPosNormal, fcPosInf | fcPosNormal));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(
      fcPosInf | fcPosNormal | fcSubnormal, fcPosNormal));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(
      fcPosNormal, fcPosInf | fcPosNormal | fcSubnormal));

  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcPosSubnormal | fcPosZero, fcPosZero));
  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcPosZero, fcPosSubnormal | fcPosZero));

  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcSubnormal | fcPosZero, fcPosZero));
  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcPosZero, fcSubnormal | fcPosZero));

  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcSubnormal | fcPosZero, fcNegZero));
  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcNegZero, fcSubnormal | fcPosZero));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegative, fcPositive));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNegative, fcPositive,
                                           /*OrderedZeroSign=*/true));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPositive, fcNegative));

  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcNegative | fcNan, fcPositive | fcNan));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(
      fcNegative | fcNan, fcPositive | fcNan, /*OrderedZeroSign=*/true));
  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcPositive | fcNan, fcNegative | fcNan));

  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcZero, fcNan));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNan, fcZero));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcZero, fcSubnormal));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcSubnormal, fcZero));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcFinite, fcFinite));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcFinite, fcInf));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcInf, fcInf));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcNegFinite, fcPosFinite));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNegFinite, fcPosFinite,
                                           /*OrderedZeroSign=*/true));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosFinite, fcNegFinite));

  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcNegFinite, fcPosFinite | fcNegZero));
  EXPECT_FALSE(
      cannotOrderStrictlyGreaterEq(fcPosFinite | fcNegZero, fcNegFinite));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcZero | fcInf, fcZero | fcInf));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcZero, fcZero | fcInf));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcZero | fcInf, fcZero));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcZero | fcInf | fcNormal, fcZero));

  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(fcPosInf, ~fcPosInf));
  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(~fcPosInf, fcPosInf));

  EXPECT_TRUE(cannotOrderStrictlyGreaterEq(fcNegInf, ~fcNegInf));
  EXPECT_FALSE(cannotOrderStrictlyGreaterEq(~fcNegInf, fcNegInf));
}

TEST(FloatingPointModeTest, cannotOrderStrictlyLess) {
  EXPECT_TRUE(cannotOrderStrictlyLess(fcNone, fcNone));
  EXPECT_FALSE(cannotOrderStrictlyLess(fcAllFlags, fcAllFlags));

  EXPECT_FALSE(cannotOrderStrictlyLess(OrderedLessThanZeroMask,
                                       OrderedLessThanZeroMask));
  EXPECT_FALSE(cannotOrderStrictlyLess(OrderedLessThanZeroMask,
                                       OrderedGreaterThanZeroMask));

  EXPECT_TRUE(cannotOrderStrictlyLess(OrderedGreaterThanZeroMask,
                                      OrderedLessThanZeroMask));

  EXPECT_TRUE(cannotOrderStrictlyLess(fcPositive, fcNegative));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcPositive, fcNegative,
                                      /*OrderedZeroSign=*/true));

  EXPECT_FALSE(cannotOrderStrictlyLess(fcNegative, fcPositive));
  EXPECT_FALSE(cannotOrderStrictlyLess(fcNegative, fcPositive,
                                       /*OrderedZeroSign=*/true));

  EXPECT_TRUE(cannotOrderStrictlyLess(fcPositive | fcNan, fcNegative));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcPositive | fcNan, fcNegative,
                                      /*OrderedZeroSign=*/true));

  EXPECT_TRUE(cannotOrderStrictlyLess(fcPositive, fcNegative | fcNan));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcPositive, fcNegative | fcNan,
                                      /*OrderedZeroSign=*/true));

  EXPECT_TRUE(cannotOrderStrictlyLess(OrderedGreaterThanZeroMask | fcNan,
                                      OrderedLessThanZeroMask));
  EXPECT_TRUE(cannotOrderStrictlyLess(OrderedGreaterThanZeroMask,
                                      OrderedLessThanZeroMask | fcNan));
  EXPECT_TRUE(cannotOrderStrictlyLess(OrderedGreaterThanZeroMask | fcNan,
                                      OrderedLessThanZeroMask | fcNan));

  EXPECT_TRUE(cannotOrderStrictlyLess(fcNan, fcNan));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcQNan, fcQNan));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcSNan, fcSNan));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcSNan, fcQNan));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcQNan, fcSNan));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcQNan, fcNan));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcSNan, fcNan));

  EXPECT_TRUE(cannotOrderStrictlyLess(fcNegInf, fcSNan));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcNegInf, fcQNan));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcNegInf, fcNan));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcSNan, fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcQNan, fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcNan, fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcNegInf, fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcNegZero, fcPosZero));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcPosZero, fcNegZero));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcPosZero, fcPosZero));
  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcPosZero, fcPosZero));

  EXPECT_TRUE(
      cannotOrderStrictlyLess(fcPosZero, fcNegZero, /*OrderedZeroSign=*/true));

  EXPECT_TRUE(cannotOrderStrictlyLess(fcZero, fcPosZero));
  EXPECT_FALSE(cannotOrderStrictlyLess(fcZero, fcPosZero, true));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcPosZero, fcZero));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcPosZero, fcZero, true));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcZero, fcNegZero));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcZero, fcNegZero, true));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcNegZero, fcZero));
  EXPECT_FALSE(cannotOrderStrictlyLess(fcNegZero, fcZero, true));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcZero, fcZero));
  EXPECT_FALSE(cannotOrderStrictlyLess(fcZero, fcZero, true));

  EXPECT_TRUE(cannotOrderStrictlyLess(fcInf, fcNan));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcNan, fcInf));

  EXPECT_TRUE(
      cannotOrderStrictlyLess(fcPosInf | fcPosZero, fcNegZero | fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyLess(
      fcPosInf | fcPosZero, fcNegZero | fcNegInf, /*OrderedZeroSign=*/true));

  EXPECT_TRUE(cannotOrderStrictlyLess(fcPosInf | fcPosZero, fcZero | fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyLess(fcPosInf | fcPosZero, fcZero | fcNegInf,
                                      /*OrderedZeroSign=*/true));

  EXPECT_FALSE(
      cannotOrderStrictlyLess(fcZero | fcNegInf, fcPosInf | fcPosZero));

  EXPECT_FALSE(
      cannotOrderStrictlyLess(fcNegInf | fcNegZero, fcZero | fcPosInf));
  EXPECT_FALSE(
      cannotOrderStrictlyLess(fcNegInf | fcNegZero, fcZero | fcPosInf, true));

  EXPECT_TRUE(cannotOrderStrictlyLess(fcZero | fcPosInf, fcNegInf | fcNegZero));
  EXPECT_TRUE(
      cannotOrderStrictlyLess(fcZero | fcPosInf, fcNegInf | fcNegZero, true));

  EXPECT_TRUE(cannotOrderStrictlyLess(fcPosInf, ~fcPosInf));
  EXPECT_FALSE(cannotOrderStrictlyLess(~fcPosInf, fcPosInf));

  EXPECT_FALSE(cannotOrderStrictlyLess(fcNegInf, ~fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyLess(~fcNegInf, fcNegInf));
}

TEST(FloatingPointModeTest, cannotOrderStrictlyLessEq) {
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcNone, fcNone));
  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcAllFlags, fcAllFlags));

  EXPECT_FALSE(cannotOrderStrictlyLessEq(OrderedLessThanZeroMask,
                                         OrderedLessThanZeroMask));
  EXPECT_FALSE(cannotOrderStrictlyLessEq(OrderedLessThanZeroMask,
                                         OrderedGreaterThanZeroMask));

  EXPECT_TRUE(cannotOrderStrictlyLessEq(OrderedGreaterThanZeroMask,
                                        OrderedLessThanZeroMask));

  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcPositive, fcNegative));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcPositive, fcNegative,
                                        /*OrderedZeroSign=*/true));

  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcNegative, fcPositive));
  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcNegative, fcPositive,
                                         /*OrderedZeroSign=*/true));

  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcPositive | fcNan, fcNegative));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcPositive | fcNan, fcNegative,
                                        /*OrderedZeroSign=*/true));

  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcPositive, fcNegative | fcNan));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcPositive, fcNegative | fcNan,
                                        /*OrderedZeroSign=*/true));

  EXPECT_TRUE(cannotOrderStrictlyLessEq(OrderedGreaterThanZeroMask | fcNan,
                                        OrderedLessThanZeroMask));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(OrderedGreaterThanZeroMask,
                                        OrderedLessThanZeroMask | fcNan));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(OrderedGreaterThanZeroMask | fcNan,
                                        OrderedLessThanZeroMask | fcNan));

  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcNan, fcNan));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcQNan, fcQNan));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcSNan, fcSNan));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcSNan, fcQNan));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcQNan, fcSNan));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcQNan, fcNan));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcSNan, fcNan));

  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcNegInf, fcSNan));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcNegInf, fcQNan));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcNegInf, fcNan));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcSNan, fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcQNan, fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcNan, fcNegInf));
  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcNegInf, fcNegInf));
  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcNegZero, fcPosZero));
  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcPosZero, fcNegZero));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcPosZero, fcNegZero,
                                        /*OrderedZeroSign=*/true));

  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcZero, fcPosZero));
  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcPosZero, fcZero));
  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcZero, fcNegZero));
  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcNegZero, fcZero));
  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcZero, fcZero));

  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcInf, fcNan));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcNan, fcInf));

  EXPECT_FALSE(
      cannotOrderStrictlyLessEq(fcPosInf | fcPosZero, fcNegZero | fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(
      fcPosInf | fcPosZero, fcNegZero | fcNegInf, /*OrderedZeroSign=*/true));

  EXPECT_FALSE(
      cannotOrderStrictlyLessEq(fcPosInf | fcPosZero, fcZero | fcNegInf));

  EXPECT_FALSE(
      cannotOrderStrictlyLessEq(fcZero | fcNegInf, fcPosInf | fcPosZero));

  EXPECT_FALSE(
      cannotOrderStrictlyLessEq(fcNegInf | fcNegZero, fcZero | fcPosInf));
  EXPECT_FALSE(
      cannotOrderStrictlyLessEq(fcZero | fcPosInf, fcNegInf | fcNegZero));

  EXPECT_TRUE(cannotOrderStrictlyLessEq(fcPosInf, ~fcPosInf));
  EXPECT_FALSE(cannotOrderStrictlyLessEq(~fcPosInf, fcPosInf));

  EXPECT_FALSE(cannotOrderStrictlyLessEq(fcNegInf, ~fcNegInf));
  EXPECT_TRUE(cannotOrderStrictlyLessEq(~fcNegInf, fcNegInf));
}
}
