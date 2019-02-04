//===- llvm/unittest/Support/DebugCounterTest.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
