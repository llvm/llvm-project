//===- llvm/unittest/ADT/StatisticTest.cpp - Statistic unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
using namespace llvm;

using OptionalStatistic = Optional<std::pair<StringRef, unsigned>>;

namespace {
#define DEBUG_TYPE "unittest"
STATISTIC(Counter, "Counts things");
STATISTIC(Counter2, "Counts other things");

#if LLVM_ENABLE_STATS
static void
extractCounters(const std::vector<std::pair<StringRef, unsigned>> &Range,
                OptionalStatistic &S1, OptionalStatistic &S2) {
  for (const auto &S : Range) {
    if (S.first == "Counter")
      S1 = S;
    if (S.first == "Counter2")
      S2 = S;
  }
}
#endif

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
  {
    const auto Range1 = GetStatistics();
    EXPECT_NE(Range1.begin(), Range1.end());
    EXPECT_EQ(Range1.begin() + 1, Range1.end());

    OptionalStatistic S1;
    OptionalStatistic S2;
    extractCounters(Range1, S1, S2);

    EXPECT_EQ(S1.hasValue(), true);
    EXPECT_EQ(S2.hasValue(), false);
  }

  // Counter2 will be registered when it's first touched.
  Counter2++;

  {
    const auto Range = GetStatistics();
    EXPECT_NE(Range.begin(), Range.end());
    EXPECT_EQ(Range.begin() + 2, Range.end());

    OptionalStatistic S1;
    OptionalStatistic S2;
    extractCounters(Range, S1, S2);

    EXPECT_EQ(S1.hasValue(), true);
    EXPECT_EQ(S2.hasValue(), true);

    EXPECT_EQ(S1->first, "Counter");
    EXPECT_EQ(S1->second, 2u);

    EXPECT_EQ(S2->first, "Counter2");
    EXPECT_EQ(S2->second, 1u);
  }
#else
  Counter2++;
  auto &Range = GetStatistics();
  EXPECT_EQ(Range.begin(), Range.end());
#endif

#if LLVM_ENABLE_STATS
  // Check that resetting the statistics works correctly.
  // It should empty the list and zero the counters.
  ResetStatistics();
  {
    auto &Range = GetStatistics();
    EXPECT_EQ(Range.begin(), Range.end());
    EXPECT_EQ(Counter, 0u);
    EXPECT_EQ(Counter2, 0u);
    OptionalStatistic S1;
    OptionalStatistic S2;
    extractCounters(Range, S1, S2);
    EXPECT_EQ(S1.hasValue(), false);
    EXPECT_EQ(S2.hasValue(), false);
  }

  // Now check that they successfully re-register and count.
  Counter++;
  Counter2++;

  {
    auto &Range = GetStatistics();
    EXPECT_EQ(Range.begin() + 2, Range.end());
    EXPECT_EQ(Counter, 1u);
    EXPECT_EQ(Counter2, 1u);

    OptionalStatistic S1;
    OptionalStatistic S2;
    extractCounters(Range, S1, S2);

    EXPECT_EQ(S1.hasValue(), true);
    EXPECT_EQ(S2.hasValue(), true);

    EXPECT_EQ(S1->first, "Counter");
    EXPECT_EQ(S1->second, 1u);

    EXPECT_EQ(S2->first, "Counter2");
    EXPECT_EQ(S2->second, 1u);
  }
#else
  // No need to test the output ResetStatistics(), there's nothing to reset so
  // we can't tell if it failed anyway.
  ResetStatistics();
#endif
}

} // end anonymous namespace
