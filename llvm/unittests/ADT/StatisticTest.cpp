//===- llvm/unittest/ADT/StatisticTest.cpp - Statistic unit tests ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {
#define DEBUG_TYPE "unittest"
STATISTIC(Counter, "Counts things");
STATISTIC(Counter2, "Counts other things");

TEST(StatisticTest, Count) {
  EnableStatistics();

  Counter = 0;
  EXPECT_EQ(Counter, 0u);
  Counter++;
  Counter++;
#if LLVM_ENABLE_STATS
  EXPECT_EQ(Counter, 2u);
#else
  EXPECT_EQ(Counter, 0u);
#endif
}

TEST(StatisticTest, Assign) {
  EnableStatistics();

  Counter = 2;
#if LLVM_ENABLE_STATS
  EXPECT_EQ(Counter, 2u);
#else
  EXPECT_EQ(Counter, 0u);
#endif
}

TEST(StatisticTest, API) {
  EnableStatistics();

  Counter = 0;
  EXPECT_EQ(Counter, 0u);
  Counter++;
  Counter++;
#if LLVM_ENABLE_STATS
  EXPECT_EQ(Counter, 2u);
#else
  EXPECT_EQ(Counter, 0u);
#endif

#if LLVM_ENABLE_STATS
  const auto Range1 = GetStatistics();
  EXPECT_NE(Range1.begin(), Range1.end());
  EXPECT_EQ(Range1.begin() + 1, Range1.end());

  Optional<std::pair<StringRef, unsigned>> S1;
  Optional<std::pair<StringRef, unsigned>> S2;
  for (const auto &S : Range1) {
    if (std::string(S.first) == "Counter")
      S1 = S;
    if (std::string(S.first) == "Counter2")
      S2 = S;
  }

  EXPECT_NE(S1.hasValue(), false);
  EXPECT_EQ(S2.hasValue(), false);

  // Counter2 will be registered when it's first touched.
  Counter2++;

  const auto Range2 = GetStatistics();
  EXPECT_NE(Range2.begin(), Range2.end());
  EXPECT_EQ(Range2.begin() + 2, Range2.end());

  S1 = None;
  S2 = None;
  for (const auto &S : Range2) {
    if (std::string(S.first) == "Counter")
      S1 = S;
    if (std::string(S.first) == "Counter2")
      S2 = S;
  }

  EXPECT_NE(S1.hasValue(), false);
  EXPECT_NE(S2.hasValue(), false);

  EXPECT_EQ(S1->first, "Counter");
  EXPECT_EQ(S1->second, 2u);

  EXPECT_EQ(S2->first, "Counter2");
  EXPECT_EQ(S2->second, 1u);
#else
  Counter2++;
  auto &Range = GetStatistics();
  EXPECT_EQ(Range.begin(), Range.end());
#endif
}

} // end anonymous namespace
