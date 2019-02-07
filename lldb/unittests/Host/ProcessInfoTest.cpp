//===-- ProcessInfoTest.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/ProcessInfo.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(ProcessInfoTest, Constructor) {
  ProcessInfo Info("foo", ArchSpec("x86_64-pc-linux"), 47);
  EXPECT_STREQ("foo", Info.GetName());
  EXPECT_EQ(ArchSpec("x86_64-pc-linux"), Info.GetArchitecture());
  EXPECT_EQ(47u, Info.GetProcessID());
}
