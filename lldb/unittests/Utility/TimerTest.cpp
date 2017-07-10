//===-- TimerTest.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
    Timer t(tcat, "");
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
    Timer t1(tcat1, "");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // Explicitly testing the same category as above.
    Timer t2(tcat1, "");
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
    Timer t1(tcat1, "");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    static Timer::Category tcat2("CAT2");
    Timer t2(tcat2, "");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  StreamString ss;
  Timer::DumpCategoryTimes(&ss);
  double seconds1, seconds2;
  ASSERT_EQ(2, sscanf(ss.GetData(), "%lf sec for CAT1%*[\n ]%lf sec for CAT2",
                      &seconds1, &seconds2))
      << "String: " << ss.GetData();
  EXPECT_LT(0.01, seconds1);
  EXPECT_GT(1, seconds1);
  EXPECT_LT(0.001, seconds2);
  EXPECT_GT(0.1, seconds2);
}
