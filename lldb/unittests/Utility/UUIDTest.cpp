//===-- UUIDTest.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/UUID.h"

using namespace lldb_private;

TEST(UUIDTest, RelationalOperators) {
  UUID empty;
  UUID a16("1234567890123456", 16);
  UUID b16("1234567890123457", 16);
  UUID a20("12345678901234567890", 20);
  UUID b20("12345678900987654321", 20);

  EXPECT_EQ(empty, empty);
  EXPECT_EQ(a16, a16);
  EXPECT_EQ(a20, a20);

  EXPECT_NE(a16, b16);
  EXPECT_NE(a20, b20);
  EXPECT_NE(a16, a20);
  EXPECT_NE(empty, a16);

  EXPECT_LT(empty, a16);
  EXPECT_LT(a16, a20);
  EXPECT_LT(a16, b16);
  EXPECT_GT(a20, b20);
}
