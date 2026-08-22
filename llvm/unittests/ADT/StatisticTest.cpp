//===- llvm/unittest/ADT/StatisticTest.cpp - Statistic unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
using namespace llvm;

using OptionalStatistic = std::optional<std::pair<StringRef, uint64_t>>;

namespace {
#define DEBUG_TYPE "unittest"
STATISTIC(Counter, "Counts things");
STATISTIC(Counter2, "Counts other things");
ALWAYS_ENABLED_STATISTIC(AlwaysCounter, "Counts things always");

#if LLVM_ENABLE_STATS
static OptionalStatistic
findStat(const std::vector<std::pair<StringRef, uint64_t>> &Stats,
         StringRef Name) {
  auto It = llvm::find_if(Stats, [&](const std::pair<StringRef, uint64_t> &S) {
    return S.first == Name;
  });
  return It != Stats.end() ? std::optional(*It) : std::nullopt;
}
#endif

TEST(StatisticTest, Count) {
  EnableStatistics();

  Counter = 0;
  EXPECT_EQ(Counter, 0ull);
  Counter++;
  Counter++;
  Counter += (std::numeric_limits<uint64_t>::max() - 3);
#if LLVM_ENABLE_STATS
  EXPECT_EQ(Counter, std::numeric_limits<uint64_t>::max() - 1);
#else
  EXPECT_EQ(Counter, UINT64_C(0));
#endif

  AlwaysCounter = 0;
  EXPECT_EQ(AlwaysCounter, 0ull);
  AlwaysCounter++;
  ++AlwaysCounter;
  EXPECT_EQ(AlwaysCounter, 2ull);
}

TEST(StatisticTest, Assign) {
  EnableStatistics();

  Counter = 2;
#if LLVM_ENABLE_STATS
  EXPECT_EQ(Counter, 2u);
#else
  EXPECT_EQ(Counter, 0u);
#endif

  AlwaysCounter = 2;
  EXPECT_EQ(AlwaysCounter, 2u);
}

TEST(StatisticTest, API) {
  EnableStatistics();
  // Reset beforehand to make sure previous tests don't effect this one.
  ResetStatistics();

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
    EXPECT_EQ(Range1.size(), 1u);

    auto S1 = findStat(Range1, "Counter");
    auto S2 = findStat(Range1, "Counter2");

    EXPECT_TRUE(S1.has_value());
    EXPECT_FALSE(S2.has_value());
  }

  // Counter2 will be registered when it's first touched.
  Counter2++;

  {
    const auto Range = GetStatistics();
    EXPECT_EQ(Range.size(), 2u);

    auto S1 = findStat(Range, "Counter");
    auto S2 = findStat(Range, "Counter2");

    ASSERT_TRUE(S1.has_value());
    EXPECT_EQ(S1->first, "Counter");
    EXPECT_EQ(S1->second, 2u);

    ASSERT_TRUE(S2.has_value());
    EXPECT_EQ(S2->first, "Counter2");
    EXPECT_EQ(S2->second, 1u);
  }
#else
  Counter2++;
  auto Range = GetStatistics();
  EXPECT_EQ(Range.begin(), Range.end());
#endif

#if LLVM_ENABLE_STATS
  // Check that resetting the statistics works correctly.
  // It should empty the list and zero the counters.
  ResetStatistics();
  {
    auto Range = GetStatistics();
    EXPECT_TRUE(Range.empty());
    EXPECT_EQ(Counter, 0u);
    EXPECT_EQ(Counter2, 0u);

    auto S1 = findStat(Range, "Counter");
    auto S2 = findStat(Range, "Counter2");
    EXPECT_FALSE(S1.has_value());
    EXPECT_FALSE(S2.has_value());
  }

  // Now check that they successfully re-register and count.
  Counter++;
  Counter2++;

  {
    auto Range = GetStatistics();
    EXPECT_EQ(Range.size(), 2u);
    EXPECT_EQ(Counter, 1u);
    EXPECT_EQ(Counter2, 1u);

    auto S1 = findStat(Range, "Counter");
    auto S2 = findStat(Range, "Counter2");

    ASSERT_TRUE(S1.has_value());
    EXPECT_EQ(S1->first, "Counter");
    EXPECT_EQ(S1->second, 1u);

    ASSERT_TRUE(S2.has_value());
    EXPECT_EQ(S2->first, "Counter2");
    EXPECT_EQ(S2->second, 1u);
  }
#else
  // No need to test the output ResetStatistics(), there's nothing to reset so
  // we can't tell if it failed anyway.
  ResetStatistics();
#endif
}

#undef DEBUG_TYPE
#define DEBUG_TYPE "local-unittest"
LOCAL_STATISTIC(LocalCounter, "Counts things locally");
LOCAL_STATISTIC(LocalCounter2, "Counts other things locally");

#undef DEBUG_TYPE
#define DEBUG_TYPE "other-unittest"
LOCAL_STATISTIC(OtherCounter, "Counts things in another debug type");

#if LLVM_ENABLE_STATS
static std::optional<StatisticVal>
findLocalStat(const std::vector<StatisticVal> &Stats, StringRef Name) {
  auto It = llvm::find_if(
      Stats, [&](const StatisticVal &S) { return S.Name == Name; });
  return It != Stats.end() ? std::optional(*It) : std::nullopt;
}
#endif

TEST(StatisticTest, LocalStatistics) {
  EnableStatistics();
  ResetStatistics();

#if LLVM_ENABLE_STATS
  // Increment a local counter before enabling local stats. This should
  // still update the global stat but not be tracked locally.
  LocalCounter++;
  {
    auto Global = GetStatistics();
    auto GS = findStat(Global, "LocalCounter");
    ASSERT_TRUE(GS.has_value());
    EXPECT_EQ(GS->second, 1u);
  }
  {
    auto Local = GetLocalStatistics();
    EXPECT_TRUE(Local.empty());
  }

  // Now enable local stats and verify the pre-enable increment isn't tracked.
  EnableLocalStatistics();

  {
    auto Local = GetLocalStatistics();
    EXPECT_TRUE(Local.empty());
  }

  // Increment local counters and verify basic functionality.
  LocalCounter++;
  LocalCounter++;
  LocalCounter2++;

  EXPECT_EQ(LocalCounter, 2u);
  EXPECT_EQ(LocalCounter2, 1u);

  {
    auto Local = GetLocalStatistics();
    EXPECT_EQ(Local.size(), 2u);

    auto S1 = findLocalStat(Local, "LocalCounter");
    auto S2 = findLocalStat(Local, "LocalCounter2");

    ASSERT_TRUE(S1.has_value());
    EXPECT_EQ(S1->DebugType, "local-unittest");
    EXPECT_EQ(S1->Value, 2u);

    ASSERT_TRUE(S2.has_value());
    EXPECT_EQ(S2->DebugType, "local-unittest");
    EXPECT_EQ(S2->Value, 1u);
  }

  // Verify that global stats are kept in sync.
  {
    auto Global = GetStatistics();
    auto GS1 = findStat(Global, "LocalCounter");
    auto GS2 = findStat(Global, "LocalCounter2");

    ASSERT_TRUE(GS1.has_value());
    EXPECT_EQ(GS1->second, 3u);

    ASSERT_TRUE(GS2.has_value());
    EXPECT_EQ(GS2->second, 1u);
  }

  // Reset local stats and verify they are cleared while globals remain.
  ResetLocalStatistics();

  {
    auto Local = GetLocalStatistics();
    EXPECT_TRUE(Local.empty());
  }

  {
    auto Global = GetStatistics();
    auto GS1 = findStat(Global, "LocalCounter");
    ASSERT_TRUE(GS1.has_value());
    EXPECT_EQ(GS1->second, 3u);
  }

  // Verify local stats work again after reset.
  LocalCounter++;
  LocalCounter2 += 5;
  {
    auto Local = GetLocalStatistics();
    EXPECT_EQ(Local.size(), 2u);

    auto S1 = findLocalStat(Local, "LocalCounter");
    auto S2 = findLocalStat(Local, "LocalCounter2");

    ASSERT_TRUE(S1.has_value());
    EXPECT_EQ(S1->Value, 1u);

    ASSERT_TRUE(S2.has_value());
    EXPECT_EQ(S2->Value, 5u);
  }

  // Global stats should reflect cumulative totals.
  {
    auto Global = GetStatistics();
    auto GS1 = findStat(Global, "LocalCounter");
    auto GS2 = findStat(Global, "LocalCounter2");

    ASSERT_TRUE(GS1.has_value());
    EXPECT_EQ(GS1->second, 4u);

    ASSERT_TRUE(GS2.has_value());
    EXPECT_EQ(GS2->second, 6u);
  }

  // Verify sorting across different debug types.
  ResetLocalStatistics();
  OtherCounter += 10;
  LocalCounter++;
  LocalCounter2 += 3;
  {
    auto Local = GetLocalStatistics();
    EXPECT_EQ(Local.size(), 3u);

    // Results should be sorted by debug type, then by name.
    EXPECT_EQ(Local[0].DebugType, "local-unittest");
    EXPECT_EQ(Local[0].Name, "LocalCounter");
    EXPECT_EQ(Local[0].Value, 1u);

    EXPECT_EQ(Local[1].DebugType, "local-unittest");
    EXPECT_EQ(Local[1].Name, "LocalCounter2");
    EXPECT_EQ(Local[1].Value, 3u);

    EXPECT_EQ(Local[2].DebugType, "other-unittest");
    EXPECT_EQ(Local[2].Name, "OtherCounter");
    EXPECT_EQ(Local[2].Value, 10u);
  }
#else
  EnableLocalStatistics();
  LocalCounter++;
  LocalCounter2++;
  auto Local = GetLocalStatistics();
  EXPECT_TRUE(Local.empty());
#endif
  ShutdownLocalStatistics();
}

} // end anonymous namespace
