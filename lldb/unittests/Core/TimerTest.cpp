//===-- TimerTest.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(_MSC_VER) && (_HAS_EXCEPTIONS == 0)
// Workaround for MSVC standard library bug, which fails to include <thread>
// when exceptions are disabled.
#include <eh.h>
#endif

#include "lldb/Core/Timer.h"
#include "gtest/gtest.h"

#include "lldb/Utility/StreamString.h"
#include <thread>

using namespace lldb_private;

TEST(TimerTest, CategoryTimes) {
  Timer::ResetCategoryTimes();
  {
    Timer t("CAT1", "");
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
    Timer t1("CAT1", "");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Timer t2("CAT1", "");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  StreamString ss;
  Timer::DumpCategoryTimes(&ss);
  double seconds;
  ASSERT_EQ(1, sscanf(ss.GetData(), "%lf sec for CAT1", &seconds));
  EXPECT_LT(0.002, seconds);
  EXPECT_GT(0.2, seconds);
}

TEST(TimerTest, CategoryTimes2) {
  Timer::ResetCategoryTimes();
  {
    Timer t1("CAT1", "");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Timer t2("CAT2", "");
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
