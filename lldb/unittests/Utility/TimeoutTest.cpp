//===-- TimeoutTest.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Timeout.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace std::chrono;

TEST(TimeoutTest, Construction) {
  EXPECT_FALSE(Timeout<std::micro>(llvm::None));
  EXPECT_TRUE(bool(Timeout<std::micro>(seconds(0))));
  EXPECT_EQ(seconds(0), *Timeout<std::micro>(seconds(0)));
  EXPECT_EQ(seconds(3), *Timeout<std::micro>(seconds(3)));
  EXPECT_TRUE(bool(Timeout<std::micro>(Timeout<std::milli>(seconds(0)))));
}

TEST(TimeoutTest, Format) {
  EXPECT_EQ("<infinite>",
            llvm::formatv("{0}", Timeout<std::milli>(llvm::None)).str());
  EXPECT_EQ("1000 ms",
            llvm::formatv("{0}", Timeout<std::milli>(seconds(1))).str());
}
