//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/TargetGroup.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/TargetGroupList.h"
#include "lldb/Target/TargetList.h"
#include "lldb/Utility/ArchSpec.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

namespace {
class TargetGroupTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
    platform_linux::PlatformLinux::Initialize();
    ArchSpec arch("x86_64-pc-linux");
    PlatformSP platform_sp =
        platform_linux::PlatformLinux::CreateInstance(true, &arch);
    ASSERT_TRUE(platform_sp);
    Platform::SetHostPlatform(platform_sp);
  }

  void TearDown() override {
    platform_linux::PlatformLinux::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }

  TargetSP CreateTarget(Debugger &debugger) {
    ArchSpec arch("x86_64-pc-linux");
    TargetSP target_sp;
    PlatformSP platform_sp;
    Status error = debugger.GetTargetList().CreateTarget(
        debugger, "", arch, eLoadDependentsNo, platform_sp, target_sp);
    EXPECT_TRUE(error.Success());
    return target_sp;
  }
};
} // namespace

TEST_F(TargetGroupTest, StoresTargetRoles) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP cpu_target_sp = CreateTarget(*debugger_sp);
  TargetSP accelerator_target_sp = CreateTarget(*debugger_sp);
  ASSERT_TRUE(cpu_target_sp);
  ASSERT_TRUE(accelerator_target_sp);

  TargetGroupList &groups = debugger_sp->GetTargetGroupList();
  TargetGroupSP group_sp = groups.CreateTargetGroup();
  ASSERT_TRUE(group_sp);

  EXPECT_TRUE(group_sp->AddTarget(cpu_target_sp, TargetRole::CPU));
  EXPECT_TRUE(
      group_sp->AddTarget(accelerator_target_sp, TargetRole::Accelerator));

  EXPECT_EQ(TargetRole::CPU, group_sp->GetRole(cpu_target_sp));
  EXPECT_EQ(TargetRole::Accelerator, group_sp->GetRole(accelerator_target_sp));
  EXPECT_EQ(std::vector<TargetSP>{cpu_target_sp},
            group_sp->GetTargets(TargetRole::CPU));
  EXPECT_EQ(std::vector<TargetSP>{accelerator_target_sp},
            group_sp->GetTargets(TargetRole::Accelerator));
  EXPECT_EQ(2U, group_sp->GetTargets().size());

  EXPECT_TRUE(group_sp->AddTarget(cpu_target_sp, TargetRole::Accelerator));
  EXPECT_EQ(TargetRole::Accelerator, group_sp->GetRole(cpu_target_sp));
  EXPECT_TRUE(group_sp->GetTargets(TargetRole::CPU).empty());
  EXPECT_EQ(2U, group_sp->GetTargets(TargetRole::Accelerator).size());

  ASSERT_EQ(1U, cpu_target_sp->GetTargetGroups().size());
  EXPECT_EQ(group_sp, cpu_target_sp->GetTargetGroups().front());
  EXPECT_EQ(1U, groups.GetNumTargetGroups());
  EXPECT_EQ(group_sp, groups.GetTargetGroupAtIndex(0));
}

TEST_F(TargetGroupTest, DeleteGroupCleansMembership) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(*debugger_sp);
  ASSERT_TRUE(target_sp);
  TargetGroupList &groups = debugger_sp->GetTargetGroupList();
  TargetGroupSP group_sp = groups.CreateTargetGroup();
  EXPECT_TRUE(group_sp->AddTarget(target_sp, TargetRole::CPU));

  EXPECT_TRUE(groups.DeleteTargetGroup(group_sp));

  EXPECT_TRUE(target_sp->GetTargetGroups().empty());
  EXPECT_FALSE(group_sp->AddTarget(target_sp, TargetRole::CPU));
  EXPECT_EQ(TargetRole::None, group_sp->GetRole(target_sp));
  EXPECT_TRUE(group_sp->GetTargets().empty());
  EXPECT_EQ(0U, groups.GetNumTargetGroups());
}

TEST_F(TargetGroupTest, TargetDestructionRemovesEveryMembership) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(*debugger_sp);
  TargetSP survivor_sp = CreateTarget(*debugger_sp);
  ASSERT_TRUE(target_sp);
  ASSERT_TRUE(survivor_sp);
  TargetGroupList &groups = debugger_sp->GetTargetGroupList();
  TargetGroupSP first_group_sp = groups.CreateTargetGroup();
  TargetGroupSP second_group_sp = groups.CreateTargetGroup();
  EXPECT_TRUE(first_group_sp->AddTarget(target_sp, TargetRole::CPU));
  EXPECT_TRUE(second_group_sp->AddTarget(target_sp, TargetRole::Accelerator));
  EXPECT_TRUE(second_group_sp->AddTarget(survivor_sp, TargetRole::CPU));

  target_sp->Destroy();

  EXPECT_TRUE(target_sp->GetTargetGroups().empty());
  EXPECT_EQ(TargetRole::None, first_group_sp->GetRole(target_sp));
  EXPECT_EQ(TargetRole::None, second_group_sp->GetRole(target_sp));
  EXPECT_EQ(TargetRole::CPU, second_group_sp->GetRole(survivor_sp));
}
