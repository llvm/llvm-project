//===-- TimerTest.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/Timer.h"
#include "gtest/gtest.h"
#include <thread>

using namespace lldb_private;

TEST(TimerTest, CategoryTimes) {
  Timer::ResetCategoryTimes();
  {
    static Timer::Category tcat("CAT1");
    Timer t(tcat, ".");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  StreamString ss;
  Timer::DumpCategoryTimes(&ss);
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
  Timer::DumpCategoryTimes(&ss);
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
  Timer::DumpCategoryTimes(&ss);
  double seconds1, seconds2;
  ASSERT_EQ(2, sscanf(ss.GetData(),
                      "%lf sec (total: %*fs; child: %*fs; count: %*d) for "
                      "CAT1%*[\n ]%lf sec for CAT2",
                      &seconds1, &seconds2))
      << "String: " << ss.GetData();
  EXPECT_LT(0.01, seconds1);
  EXPECT_GT(1, seconds1);
  EXPECT_LT(0.001, seconds2);
  EXPECT_GT(0.1, seconds2);
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
  // Example output:
  // 0.105202764 sec (total: 0.132s; child: 0.027s; count: 1) for CAT1
  // 0.026772798 sec (total: 0.027s; child: 0.000s; count: 2) for CAT2
  StreamString ss;
  Timer::DumpCategoryTimes(&ss);
  double seconds1, total1, child1, seconds2;
  int count1, count2;
  ASSERT_EQ(
      6, sscanf(ss.GetData(),
                "%lf sec (total: %lfs; child: %lfs; count: %d) for CAT1%*[\n\r ]"
                "%lf sec (total: %*fs; child: %*fs; count: %d) for CAT2",
                &seconds1, &total1, &child1, &count1, &seconds2, &count2))
      << "String: " << ss.GetData();
  EXPECT_NEAR(total1 - child1, seconds1, 0.002);
  EXPECT_EQ(1, count1);
  EXPECT_NEAR(child1, seconds2, 0.002);
  EXPECT_EQ(2, count2);
}
