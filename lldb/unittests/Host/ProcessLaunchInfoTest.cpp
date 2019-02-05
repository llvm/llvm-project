//===-- ProcessLaunchInfoTest.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
