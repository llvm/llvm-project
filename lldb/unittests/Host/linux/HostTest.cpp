//===-- HostTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Host.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/ProcessInfo.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {
class HostTest : public testing::Test {
public:
  static void SetUpTestCase() {
    FileSystem::Initialize();
    HostInfo::Initialize();
  }
  static void TearDownTestCase() {
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};
} // namespace

TEST_F(HostTest, GetProcessInfo) {
  llvm::Triple triple = HostInfo::GetTargetTriple();

  ASSERT_TRUE(
      (triple.getOS() == llvm::Triple::OSType::Linux) ||
      (triple.hasEnvironment() &&
       triple.getEnvironment() == llvm::Triple::EnvironmentType::Android));

  ProcessInstanceInfo Info;
  ASSERT_FALSE(Host::GetProcessInfo(0, Info));

  ASSERT_TRUE(Host::GetProcessInfo(getpid(), Info));

  ASSERT_TRUE(Info.ProcessIDIsValid());
  EXPECT_EQ(lldb::pid_t(getpid()), Info.GetProcessID());

  ASSERT_TRUE(Info.ParentProcessIDIsValid());
  EXPECT_EQ(lldb::pid_t(getppid()), Info.GetParentProcessID());

  ASSERT_TRUE(Info.ProcessGroupIDIsValid());
  EXPECT_EQ(lldb::pid_t(getpgrp()), Info.GetProcessGroupID());

  ASSERT_TRUE(Info.ProcessSessionIDIsValid());
  EXPECT_EQ(lldb::pid_t(getsid(getpid())), Info.GetProcessSessionID());

  ASSERT_TRUE(Info.EffectiveUserIDIsValid());
  EXPECT_EQ(geteuid(), Info.GetEffectiveUserID());

  ASSERT_TRUE(Info.EffectiveGroupIDIsValid());
  EXPECT_EQ(getegid(), Info.GetEffectiveGroupID());

  ASSERT_TRUE(Info.UserIDIsValid());
  EXPECT_EQ(geteuid(), Info.GetUserID());

  ASSERT_TRUE(Info.GroupIDIsValid());
  EXPECT_EQ(getegid(), Info.GetGroupID());

  EXPECT_TRUE(Info.GetArchitecture().IsValid());
  EXPECT_EQ(HostInfo::GetArchitecture(HostInfo::eArchKindDefault),
            Info.GetArchitecture());
  // Test timings
  /*
   * This is flaky in the buildbots on all archs
  ASSERT_TRUE(Host::GetProcessInfo(getpid(), Info));
  ProcessInstanceInfo::timespec user_time = Info.GetUserTime();
  static volatile unsigned u = 0;
  for (unsigned i = 0; i < 10'000'000; i++) {
    u = i;
  }
  ASSERT_TRUE(Host::GetProcessInfo(getpid(), Info));
  ProcessInstanceInfo::timespec next_user_time = Info.GetUserTime();
  ASSERT_TRUE(user_time.tv_sec < next_user_time.tv_sec ||
              user_time.tv_usec < next_user_time.tv_usec);
  */
}
