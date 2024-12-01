//===- unittests/TimerTest.cpp - Timer tests ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Timer.h"
#include "gtest/gtest.h"

#if _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

using namespace llvm;

namespace {

// FIXME: Put this somewhere in Support, it's also used in LockFileManager.
void SleepMS() {
#if _WIN32
  Sleep(1);
#else
  struct timespec Interval;
  Interval.tv_sec = 0;
  Interval.tv_nsec = 1000000;
#if defined(__MVS__)
  long Microseconds = (Interval.tv_nsec + 999) / 1000;
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

} // end anon namespace
