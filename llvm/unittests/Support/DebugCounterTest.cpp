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
TEST(DebugCounterTest, Basic) {
  DEBUG_COUNTER(TestCounter, "test-counter", "Counter used for unit test");

  EXPECT_FALSE(DebugCounter::isCounterSet(TestCounter));
  auto DC = &DebugCounter::instance();
  DC->push_back("test-counter=1:3-5:78:79:89:100-102:150");

  EXPECT_TRUE(DebugCounter::isCounterSet(TestCounter));

  SmallVector<unsigned> Res;
  for (unsigned Idx = 0; Idx < 200; Idx++) {
    if (DebugCounter::shouldExecute(TestCounter))
      Res.push_back(Idx);
  }

  SmallVector<unsigned> Expected = {1, 3, 4, 5, 78, 79, 89, 100, 101, 102, 150};
  EXPECT_EQ(Expected, Res);

  std::string Str;
  llvm::raw_string_ostream OS(Str);
  DC->print(OS);
  EXPECT_TRUE(StringRef(Str).contains("{200,1:3-5:78:79:89:100-102:150}"));
}

#endif
