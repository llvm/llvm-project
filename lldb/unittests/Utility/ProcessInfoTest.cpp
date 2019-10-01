//===-- ProcessInfoTest.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/ProcessInfo.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(ProcessInfoTest, Constructor) {
  ProcessInfo Info("foo", ArchSpec("x86_64-pc-linux"), 47);
  EXPECT_STREQ("foo", Info.GetName());
  EXPECT_EQ(ArchSpec("x86_64-pc-linux"), Info.GetArchitecture());
  EXPECT_EQ(47u, Info.GetProcessID());
}
