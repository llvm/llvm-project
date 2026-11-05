//===- FeatureGuardParserTest.cpp - Feature guard parser tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/FeatureGuardParser.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "gtest/gtest.h"

using namespace llvm;

static FeatureGuardAlternatives parseGuard(StringRef Text) {
  return FeatureGuardParser(
             Text, [](const Twine &Message) { PrintFatalError(Message); })
      .parse();
}

TEST(FeatureGuardParserTest, EmptyGuard) {
  EXPECT_EQ(parseGuard(""), FeatureGuardAlternatives{});
  EXPECT_EQ(parseGuard("   "), FeatureGuardAlternatives{});
}

TEST(FeatureGuardParserTest, ConjunctionAndDisjunction) {
  EXPECT_EQ(parseGuard("a,b|c"), (FeatureGuardAlternatives{{"a", "b"}, {"c"}}));
  EXPECT_EQ(parseGuard("a|b,c"), (FeatureGuardAlternatives{{"a"}, {"b", "c"}}));
}

TEST(FeatureGuardParserTest, ParenthesesDistributeConjunction) {
  EXPECT_EQ(parseGuard("a,(b|c)"),
            (FeatureGuardAlternatives{{"a", "b"}, {"a", "c"}}));
  EXPECT_EQ(parseGuard("(a|b),(c|d)"),
            (FeatureGuardAlternatives{
                {"a", "c"}, {"a", "d"}, {"b", "c"}, {"b", "d"}}));
}

TEST(FeatureGuardParserTest, WhitespaceAndDuplicates) {
  EXPECT_EQ(parseGuard(" ( a | b ) , c "),
            (FeatureGuardAlternatives{{"a", "c"}, {"b", "c"}}));
  EXPECT_EQ(parseGuard("a,a|a"), (FeatureGuardAlternatives{{"a"}}));
}

TEST(FeatureGuardParserDeathTest, InvalidExpressions) {
  RecordKeeper Records;
  Record Owner("test", {}, Records);
  auto ReportError = [&](const Twine &Message) {
    PrintFatalError(&Owner, Message);
  };
  EXPECT_EXIT(FeatureGuardParser("a|", ReportError).parse(),
              testing::ExitedWithCode(1), "expected feature in target guard");
  EXPECT_EXIT(FeatureGuardParser("(a|b", ReportError).parse(),
              testing::ExitedWithCode(1), "unclosed target guard expression");
  EXPECT_EXIT(FeatureGuardParser("a)b", ReportError).parse(),
              testing::ExitedWithCode(1), "invalid target guard expression");
  EXPECT_EXIT(FeatureGuardParser("a\tb", ReportError).parse(),
              testing::ExitedWithCode(1), "invalid target guard expression");
  EXPECT_EXIT(FeatureGuardParser("a\n", ReportError).parse(),
              testing::ExitedWithCode(1), "invalid target guard expression");
  EXPECT_EXIT(FeatureGuardParser("\ra", ReportError).parse(),
              testing::ExitedWithCode(1), "expected feature in target guard");
  EXPECT_EXIT(FeatureGuardParser("a \t|b", ReportError).parse(),
              testing::ExitedWithCode(1), "invalid target guard expression");
}
