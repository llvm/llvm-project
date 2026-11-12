//===-- TimerTest.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Timer.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"
#include "gtest/gtest.h"
#include <optional>
#include <thread>

using namespace lldb_private;

namespace {
struct CategoryStats {
  double seconds;
  double total;
  double child;
  int count;
};

/// Finds the line describing \p category in a DumpCategoryTimes() dump and
/// parses its statistics. A line looks like:
///   0.105202764 sec (total: 0.132s; child: 0.027s; count: 1) for CAT1
std::optional<CategoryStats> ParseCategory(llvm::StringRef dump,
                                           llvm::StringRef category) {
  llvm::Regex line_pattern(R"(^([0-9.]+) sec \(total: ([0-9.]+)s; )"
                           R"(child: ([0-9.]+)s; count: ([0-9]+)\) for (.+)$)");
  for (llvm::StringRef line : llvm::split(dump, '\n')) {
    llvm::SmallVector<llvm::StringRef, 6> matches;
    if (!line_pattern.match(line.trim(), &matches))
      continue;
    if (matches[5] != category)
      continue;
    CategoryStats stats;
    if (matches[1].getAsDouble(stats.seconds) ||
        matches[2].getAsDouble(stats.total) ||
        matches[3].getAsDouble(stats.child) ||
        matches[4].getAsInteger(10, stats.count))
      return std::nullopt;
    return stats;
  }
  return std::nullopt;
}
} // namespace

TEST(TimerTest, CategoryTimes) {
  Timer::ResetCategoryTimes();
  {
    static Timer::Category tcat("CAT1");
    Timer t(tcat, ".");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  StreamString ss;
  Timer::DumpCategoryTimes(ss);
  double seconds;
  ASSERT_EQ(1, sscanf(ss.GetData(), "%lf sec for CAT1", &seconds));
  EXPECT_LT(0.001, seconds);
  EXPECT_GT(0.1, seconds);
}

TEST(TimerTest, CategoryTimesNested) {
  Timer::ResetCategoryTimes();
  {
    static Timer::Category tcat1("CAT1");
    Timer t1(tcat1, ".");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // Explicitly testing the same category as above.
    Timer t2(tcat1, ".");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  StreamString ss;
  Timer::DumpCategoryTimes(ss);
  double seconds;
  // It should only appear once.
  ASSERT_EQ(ss.GetString().count("CAT1"), 1U);
  ASSERT_EQ(1, sscanf(ss.GetData(), "%lf sec for CAT1", &seconds));
  EXPECT_LT(0.002, seconds);
  EXPECT_GT(0.2, seconds);
}

TEST(TimerTest, CategoryTimes2) {
  Timer::ResetCategoryTimes();
  {
    static Timer::Category tcat1("CAT1");
    Timer t1(tcat1, ".");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    static Timer::Category tcat2("CAT2");
    Timer t2(tcat2, ".");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  StreamString ss;
  Timer::DumpCategoryTimes(ss);
  std::optional<CategoryStats> cat1 = ParseCategory(ss.GetString(), "CAT1");
  std::optional<CategoryStats> cat2 = ParseCategory(ss.GetString(), "CAT2");
  ASSERT_TRUE(cat1.has_value()) << "String: " << ss.GetData();
  ASSERT_TRUE(cat2.has_value()) << "String: " << ss.GetData();
  EXPECT_LT(0.01, cat1->seconds);
  EXPECT_GT(1, cat1->seconds);
  EXPECT_LT(0.001, cat2->seconds);
  EXPECT_GT(0.1, cat2->seconds);
}

TEST(TimerTest, CategoryTimesStats) {
  Timer::ResetCategoryTimes();
  {
    static Timer::Category tcat1("CAT1");
    Timer t1(tcat1, ".");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    static Timer::Category tcat2("CAT2");
    {
      Timer t2(tcat2, ".");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    {
      Timer t3(tcat2, ".");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  StreamString ss;
  Timer::DumpCategoryTimes(ss);
  std::optional<CategoryStats> cat1 = ParseCategory(ss.GetString(), "CAT1");
  std::optional<CategoryStats> cat2 = ParseCategory(ss.GetString(), "CAT2");
  ASSERT_TRUE(cat1.has_value()) << "String: " << ss.GetData();
  ASSERT_TRUE(cat2.has_value()) << "String: " << ss.GetData();
  EXPECT_NEAR(cat1->total - cat1->child, cat1->seconds, 0.002);
  EXPECT_EQ(1, cat1->count);
  EXPECT_NEAR(cat1->child, cat2->seconds, 0.002);
  EXPECT_EQ(2, cat2->count);
}
