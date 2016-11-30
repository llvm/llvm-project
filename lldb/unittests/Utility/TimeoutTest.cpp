//===-- TimeoutTest.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Timeout.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace std::chrono;

TEST(TimeoutTest, Construction) {
  ASSERT_FALSE(Timeout<std::micro>(llvm::None));
  ASSERT_TRUE(bool(Timeout<std::micro>(seconds(0))));
  ASSERT_EQ(seconds(0), *Timeout<std::micro>(seconds(0)));
  ASSERT_EQ(seconds(3), *Timeout<std::micro>(seconds(3)));
  ASSERT_TRUE(bool(Timeout<std::micro>(Timeout<std::milli>(seconds(0)))));
}
