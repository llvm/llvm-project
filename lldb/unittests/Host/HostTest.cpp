//===-- HostTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Host.h"
#include "lldb/Utility/ProcessInfo.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace llvm;

TEST(Host, WaitStatusFormat) {
  EXPECT_EQ("W01", formatv("{0:g}", WaitStatus{WaitStatus::Exit, 1}).str());
  EXPECT_EQ("X02", formatv("{0:g}", WaitStatus{WaitStatus::Signal, 2}).str());
  EXPECT_EQ("S03", formatv("{0:g}", WaitStatus{WaitStatus::Stop, 3}).str());
  EXPECT_EQ("Exited with status 4",
            formatv("{0}", WaitStatus{WaitStatus::Exit, 4}).str());
}

TEST(Host, GetEnvironment) {
  putenv(const_cast<char *>("LLDB_TEST_ENVIRONMENT_VAR=Host::GetEnvironment"));
  ASSERT_EQ("Host::GetEnvironment",
            Host::GetEnvironment().lookup("LLDB_TEST_ENVIRONMENT_VAR"));
}

TEST(Host, ProcessInstanceInfoCumulativeUserTimeIsValid) {
  ProcessInstanceInfo info;
  info.SetCumulativeUserTime(ProcessInstanceInfo::timespec{0, 0});
  EXPECT_FALSE(info.CumulativeUserTimeIsValid());
  info.SetCumulativeUserTime(ProcessInstanceInfo::timespec{0, 1});
  EXPECT_TRUE(info.CumulativeUserTimeIsValid());
  info.SetCumulativeUserTime(ProcessInstanceInfo::timespec{1, 0});
  EXPECT_TRUE(info.CumulativeUserTimeIsValid());
}

TEST(Host, ProcessInstanceInfoCumulativeSystemTimeIsValid) {
  ProcessInstanceInfo info;
  info.SetCumulativeSystemTime(ProcessInstanceInfo::timespec{0, 0});
  EXPECT_FALSE(info.CumulativeSystemTimeIsValid());
  info.SetCumulativeSystemTime(ProcessInstanceInfo::timespec{0, 1});
  EXPECT_TRUE(info.CumulativeSystemTimeIsValid());
  info.SetCumulativeSystemTime(ProcessInstanceInfo::timespec{1, 0});
  EXPECT_TRUE(info.CumulativeSystemTimeIsValid());
}