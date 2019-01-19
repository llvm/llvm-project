//===-- BreakpointIDTest.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Breakpoint/BreakpointID.h"
#include "lldb/Utility/Status.h"

#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;

TEST(BreakpointIDTest, StringIsBreakpointName) {
  Status E;
  EXPECT_FALSE(BreakpointID::StringIsBreakpointName("1breakpoint", E));
  EXPECT_FALSE(BreakpointID::StringIsBreakpointName("-", E));
  EXPECT_FALSE(BreakpointID::StringIsBreakpointName("", E));
  EXPECT_FALSE(BreakpointID::StringIsBreakpointName("3.4", E));

  EXPECT_TRUE(BreakpointID::StringIsBreakpointName("_", E));
  EXPECT_TRUE(BreakpointID::StringIsBreakpointName("a123", E));
  EXPECT_TRUE(BreakpointID::StringIsBreakpointName("test", E));
}
