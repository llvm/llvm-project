//===-- BreakpointIDTest.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Breakpoint/BreakpointID.h"
#include "lldb/Core/Error.h"

#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;

TEST(BreakpointIDTest, StringIsBreakpointName) {
  Error E;
  EXPECT_FALSE(BreakpointID::StringIsBreakpointName("1breakpoint", E));
  EXPECT_FALSE(BreakpointID::StringIsBreakpointName("-", E));
  EXPECT_FALSE(BreakpointID::StringIsBreakpointName("", E));
  EXPECT_FALSE(BreakpointID::StringIsBreakpointName("3.4", E));

  EXPECT_TRUE(BreakpointID::StringIsBreakpointName("_", E));
  EXPECT_TRUE(BreakpointID::StringIsBreakpointName("a123", E));
  EXPECT_TRUE(BreakpointID::StringIsBreakpointName("test", E));
}
