//===-- HostTest.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {
class HostTest : public testing::Test {
public:
  static void SetUpTestCase() { HostInfo::Initialize(); }
  static void TearDownTestCase() { HostInfo::Terminate(); }
};
} // namespace

TEST_F(HostTest, GetProcessInfo) {
  ProcessInstanceInfo Info;
  ASSERT_FALSE(Host::GetProcessInfo(0, Info));

  ASSERT_TRUE(Host::GetProcessInfo(getpid(), Info));

  ASSERT_TRUE(Info.ProcessIDIsValid());
  EXPECT_EQ(lldb::pid_t(getpid()), Info.GetProcessID());

  ASSERT_TRUE(Info.ParentProcessIDIsValid());
  EXPECT_EQ(lldb::pid_t(getppid()), Info.GetParentProcessID());

  ASSERT_TRUE(Info.EffectiveUserIDIsValid());
  EXPECT_EQ(geteuid(), Info.GetEffectiveUserID());

  ASSERT_TRUE(Info.EffectiveGroupIDIsValid());
  EXPECT_EQ(getegid(), Info.GetEffectiveGroupID());

  ASSERT_TRUE(Info.UserIDIsValid());
  EXPECT_EQ(geteuid(), Info.GetUserID());

  ASSERT_TRUE(Info.GroupIDIsValid());
  EXPECT_EQ(getegid(), Info.GetGroupID());
}
