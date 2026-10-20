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

#include <cerrno>
#include <sys/resource.h>

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
  ProcessInstanceInfo Info;

  ASSERT_FALSE(Host::GetProcessInfo(LLDB_INVALID_PROCESS_ID, Info));

  ASSERT_TRUE(Host::GetProcessInfo(getpid(), Info));

  EXPECT_TRUE(Info.ProcessIDIsValid());
  EXPECT_EQ(lldb::pid_t(getpid()), Info.GetProcessID());

  EXPECT_TRUE(Info.ParentProcessIDIsValid());
  EXPECT_EQ(lldb::pid_t(getppid()), Info.GetParentProcessID());

  // Not currently set on apple systems.
#ifndef __APPLE__
  EXPECT_TRUE(Info.ProcessGroupIDIsValid());
  EXPECT_EQ(lldb::pid_t(getpgrp()), Info.GetProcessGroupID());

  EXPECT_TRUE(Info.ProcessSessionIDIsValid());
  EXPECT_EQ(lldb::pid_t(getsid(getpid())), Info.GetProcessSessionID());
#endif

  EXPECT_TRUE(Info.EffectiveUserIDIsValid());
  EXPECT_EQ(geteuid(), Info.GetEffectiveUserID());

  EXPECT_TRUE(Info.EffectiveGroupIDIsValid());
  EXPECT_EQ(getegid(), Info.GetEffectiveGroupID());

  EXPECT_TRUE(Info.UserIDIsValid());
  EXPECT_EQ(geteuid(), Info.GetUserID());

  EXPECT_TRUE(Info.GroupIDIsValid());
  EXPECT_EQ(getegid(), Info.GetGroupID());

  // Unexpected value on Apple x86_64
#ifndef __APPLE__
  EXPECT_TRUE(Info.GetArchitecture().IsValid());
  EXPECT_EQ(HostInfo::GetArchitecture(HostInfo::eArchKindDefault),
            Info.GetArchitecture());
#endif

  // Test timings
  // In some sense this is a pretty trivial test. What it is trying to
  // accomplish is just to validate that these values are never decreasing
  // which would be unambiguously wrong. We can not reliably show them
  // to be always increasing because the microsecond granularity means that,
  // with hardware variations the number of loop iterations need to always
  // be increasing for faster and faster machines.
  ASSERT_TRUE(Host::GetProcessInfo(getpid(), Info));
  ProcessInstanceInfo::timespec user_time = Info.GetUserTime();
  static volatile unsigned u = 0;
  for (unsigned i = 0; i < 10'000'000; i++) {
    u += i;
  }
  ASSERT_TRUE(u > 0);
  ASSERT_TRUE(Host::GetProcessInfo(getpid(), Info));
  ProcessInstanceInfo::timespec next_user_time = Info.GetUserTime();
  ASSERT_TRUE(user_time.tv_sec <= next_user_time.tv_sec ||
              user_time.tv_usec <= next_user_time.tv_usec);
}

// Only linux currently sets these.
#ifdef __linux__
TEST_F(HostTest, GetProcessInfoSetsPriority) {
  ProcessInstanceInfo Info;
  struct rlimit rlim;
  EXPECT_EQ(getrlimit(RLIMIT_NICE, &rlim), 0);
  // getpriority can return -1 so we zero errno first
  errno = 0;
  int prio = getpriority(PRIO_PROCESS, 0);
  ASSERT_TRUE((prio < 0 && errno == 0) || prio >= 0);
  ASSERT_TRUE(Host::GetProcessInfo(getpid(), Info));
  ASSERT_EQ(Info.GetPriorityValue(), prio);
  // If we can't raise our nice level then this test can't be performed.
  int max_incr = PRIO_MAX - rlim.rlim_cur;
  if (max_incr < prio) {
    EXPECT_EQ(setpriority(PRIO_PROCESS, PRIO_PROCESS, prio - 1), 0);
    ASSERT_TRUE(Host::GetProcessInfo(getpid(), Info));
    ASSERT_TRUE(Info.GetPriorityValue().has_value());
    ASSERT_EQ(Info.GetPriorityValue().value(), prio - 1);
    EXPECT_EQ(setpriority(PRIO_PROCESS, PRIO_PROCESS, prio), 0);
  }
  ASSERT_TRUE(Info.IsZombie().has_value());
  ASSERT_FALSE(Info.IsZombie().value());

  const llvm::VersionTuple host_version = HostInfo::GetOSVersion();
  ASSERT_FALSE(host_version.empty());
  if (host_version >= llvm::VersionTuple(4, 15, 0)) {
    ASSERT_TRUE(Info.IsCoreDumping().has_value());
    ASSERT_FALSE(Info.IsCoreDumping().value());
  } else {
    ASSERT_FALSE(Info.IsCoreDumping().has_value());
  }
}
#endif
