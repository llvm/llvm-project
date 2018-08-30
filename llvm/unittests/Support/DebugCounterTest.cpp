//===- llvm/unittest/Support/DebugCounterTest.cpp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DebugCounter.h"
#include "gtest/gtest.h"

#include <string>
using namespace llvm;

#ifndef NDEBUG
TEST(DebugCounterTest, CounterCheck) {
  DEBUG_COUNTER(TestCounter, "test-counter", "Counter used for unit test");

  EXPECT_FALSE(DebugCounter::isCounterSet(TestCounter));

  auto DC = &DebugCounter::instance();
  DC->push_back("test-counter-skip=1");
  DC->push_back("test-counter-count=3");

  EXPECT_TRUE(DebugCounter::isCounterSet(TestCounter));

  EXPECT_EQ(0, DebugCounter::getCounterValue(TestCounter));
  EXPECT_FALSE(DebugCounter::shouldExecute(TestCounter));

  EXPECT_EQ(1, DebugCounter::getCounterValue(TestCounter));
  EXPECT_TRUE(DebugCounter::shouldExecute(TestCounter));

  DebugCounter::setCounterValue(TestCounter, 3);
  EXPECT_TRUE(DebugCounter::shouldExecute(TestCounter));
  EXPECT_FALSE(DebugCounter::shouldExecute(TestCounter));

  DebugCounter::setCounterValue(TestCounter, 100);
  EXPECT_FALSE(DebugCounter::shouldExecute(TestCounter));
}
#endif
