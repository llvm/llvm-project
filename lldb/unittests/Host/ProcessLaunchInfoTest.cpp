//===-- ProcessLaunchInfoTest.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/ProcessLaunchInfo.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

TEST(ProcessLaunchInfoTest, Constructor) {
  ProcessLaunchInfo Info(FileSpec("/stdin"), FileSpec("/stdout"),
                         FileSpec("/stderr"), FileSpec("/wd"),
                         eLaunchFlagStopAtEntry);
  EXPECT_EQ(FileSpec("/stdin"),
            Info.GetFileActionForFD(STDIN_FILENO)->GetFileSpec());
  EXPECT_EQ(FileSpec("/stdout"),
            Info.GetFileActionForFD(STDOUT_FILENO)->GetFileSpec());
  EXPECT_EQ(FileSpec("/stderr"),
            Info.GetFileActionForFD(STDERR_FILENO)->GetFileSpec());
  EXPECT_EQ(FileSpec("/wd"), Info.GetWorkingDirectory());
  EXPECT_EQ(eLaunchFlagStopAtEntry, Info.GetFlags().Get());
}
