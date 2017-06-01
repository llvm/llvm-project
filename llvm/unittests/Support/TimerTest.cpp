//===- unittests/TimerTest.cpp - Timer tests ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Timer.h"
#include "gtest/gtest.h"

#if LLVM_ON_WIN32
#include <windows.h>
#else
#include <time.h>
#endif

using namespace llvm;

namespace {

// FIXME: Put this somewhere in Support, it's also used in LockFileManager.
void SleepMS() {
#if LLVM_ON_WIN32
  Sleep(1);
#else
  struct timespec Interval;
  Interval.tv_sec = 0;
  Interval.tv_nsec = 1000000;
  nanosleep(&Interval, nullptr);
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

  EXPECT_TRUE(TR1 < TR2);
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
