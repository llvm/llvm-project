//===- unittests/TimerTest.cpp - Timer tests ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Timer.h"
#include "llvm/Support/CommandLine.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#if _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

using namespace llvm;

cl::opt<unsigned> &minPrintTime();

namespace {

// FIXME: Put this somewhere in Support, it's also used in LockFileManager.
void SleepMS(int ms = 1) {
#if _WIN32
  Sleep(ms);
#else
  struct timespec Interval;
  Interval.tv_sec = ms / 1000;
  Interval.tv_nsec = 1000000 * (ms % 1000);
#if defined(__MVS__)
  long Microseconds = (Interval.tv_nsec + Interval.tv_sec * 1000 + 999) / 1000;
  usleep(Microseconds);
#else
  nanosleep(&Interval, nullptr);
#endif
#endif
}

TEST(Timer, Additivity) {
  Timer T1("T1", "T1");

  EXPECT_TRUE(T1.isInitialized());

  T1.startTimer();
  T1.stopTimer();
  auto TR1 = T1.getTotalTime();

  T1.startTimer();
  SleepMS();
  T1.stopTimer();
  auto TR2 = T1.getTotalTime();

  EXPECT_LT(TR1, TR2);
}

TEST(Timer, CheckIfTriggered) {
  Timer T1("T1", "T1");

  EXPECT_FALSE(T1.hasTriggered());
  T1.startTimer();
  EXPECT_TRUE(T1.hasTriggered());
  T1.stopTimer();
  EXPECT_TRUE(T1.hasTriggered());

  T1.clear();
  EXPECT_FALSE(T1.hasTriggered());
}

TEST(Timer, TimerGroupTimerDestructed) {
  testing::internal::CaptureStderr();

  {
    TimerGroup TG("tg", "desc");
    {
      Timer T1("T1", "T1", TG);
      T1.startTimer();
      T1.stopTimer();
    }
    EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
    testing::internal::CaptureStderr();
  }
  EXPECT_FALSE(testing::internal::GetCapturedStderr().empty());
}

TEST(Timer, MinTimerFlag) {
  testing::internal::CaptureStderr();

  Timer T1("T1", "T1");
  Timer T2("T2", "T2");

  minPrintTime().setValue(2);

  T1.startTimer();
  T2.startTimer();

  SleepMS(1000);
  T1.stopTimer();

  SleepMS(2000);
  T2.stopTimer();

  TimerGroup::printAll(llvm::errs());
  std::string stderr = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr, testing::HasSubstr("T2"));
  EXPECT_THAT(stderr, testing::Not(testing::HasSubstr("T1")));

  testing::internal::CaptureStderr();

  TimerGroup::printAllJSONValues(llvm::errs(), "");
  stderr = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr, testing::HasSubstr("T2.wall"));
  EXPECT_THAT(stderr, testing::Not(testing::HasSubstr("T1.wall")));
}

} // namespace
