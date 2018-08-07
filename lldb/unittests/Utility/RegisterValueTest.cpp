//===-- RegisterValueTest.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/RegisterValue.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(RegisterValueTest, GetSet8) {
  RegisterValue R8(uint8_t(47));
  EXPECT_EQ(47u, R8.GetAsUInt8());
  R8 = uint8_t(42);
  EXPECT_EQ(42u, R8.GetAsUInt8());
  EXPECT_EQ(42u, R8.GetAsUInt16());
  EXPECT_EQ(42u, R8.GetAsUInt32());
  EXPECT_EQ(42u, R8.GetAsUInt64());
}
