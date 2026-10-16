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
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/TargetGroupList.h"
#include "lldb/Target/TargetList.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/Testing/Support/Error.h"
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
  TargetGroupSP group_sp =
      groups.CreateTargetGroup("CPU and accelerator targets");
  ASSERT_TRUE(group_sp);
  EXPECT_EQ("CPU and accelerator targets", group_sp->GetDescription());

  EXPECT_THAT_ERROR(group_sp->AddTarget(cpu_target_sp, TargetGroup::Role::CPU),
                    llvm::Succeeded());
  EXPECT_THAT_ERROR(group_sp->AddTarget(accelerator_target_sp,
                                        TargetGroup::Role::Accelerator),
                    llvm::Succeeded());

  EXPECT_EQ(std::optional(TargetGroup::Role::CPU),
            group_sp->GetRole(cpu_target_sp));
  EXPECT_EQ(std::optional(TargetGroup::Role::Accelerator),
            group_sp->GetRole(accelerator_target_sp));
  EXPECT_EQ(std::vector<TargetSP>{cpu_target_sp},
            group_sp->GetTargets(TargetGroup::Role::CPU));
  EXPECT_EQ(std::vector<TargetSP>{accelerator_target_sp},
            group_sp->GetTargets(TargetGroup::Role::Accelerator));
  EXPECT_EQ(2U, group_sp->GetTargets().size());

  EXPECT_THAT_ERROR(group_sp->AddTarget(cpu_target_sp, TargetGroup::Role::CPU),
                    llvm::Succeeded());
  EXPECT_THAT_ERROR(
      group_sp->AddTarget(cpu_target_sp, TargetGroup::Role::Accelerator),
      llvm::FailedWithMessage(
          "target is already in the group with a different role"));
  EXPECT_EQ(std::optional(TargetGroup::Role::CPU),
            group_sp->GetRole(cpu_target_sp));
  EXPECT_EQ(1U, group_sp->GetTargets(TargetGroup::Role::CPU).size());
  EXPECT_EQ(1U, group_sp->GetTargets(TargetGroup::Role::Accelerator).size());

  ASSERT_EQ(1U, cpu_target_sp->GetTargetGroups().size());
  EXPECT_EQ(group_sp, cpu_target_sp->GetTargetGroups().front());
  EXPECT_EQ(std::vector<TargetGroupSP>{group_sp}, groups.GetGroups());
}

TEST_F(TargetGroupTest, DeleteGroupReleasesDebuggerOwnership) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(*debugger_sp);
  ASSERT_TRUE(target_sp);
  TargetGroupList &groups = debugger_sp->GetTargetGroupList();
  TargetGroupSP group_sp = groups.CreateTargetGroup("related targets");
  EXPECT_THAT_ERROR(group_sp->AddTarget(target_sp, TargetGroup::Role::CPU),
                    llvm::Succeeded());

  EXPECT_TRUE(groups.DeleteTargetGroup(group_sp));
  EXPECT_TRUE(groups.GetGroups().empty());
  ASSERT_EQ(1U, target_sp->GetTargetGroups().size());
  EXPECT_EQ(group_sp, target_sp->GetTargetGroups().front());

  TargetGroupWP group_wp = group_sp;
  group_sp.reset();
  EXPECT_FALSE(group_wp.expired());
  ASSERT_EQ(1U, target_sp->GetTargetGroups().size());
  group_sp = target_sp->GetTargetGroups().front();
  EXPECT_TRUE(group_sp->RemoveTarget(target_sp));
  group_sp.reset();
  EXPECT_TRUE(group_wp.expired());
  EXPECT_TRUE(target_sp->GetTargetGroups().empty());
}

TEST_F(TargetGroupTest, TargetDestructionRemovesEveryMembership) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(*debugger_sp);
  TargetSP survivor_sp = CreateTarget(*debugger_sp);
  ASSERT_TRUE(target_sp);
  ASSERT_TRUE(survivor_sp);
  TargetGroupList &groups = debugger_sp->GetTargetGroupList();
  TargetGroupSP first_group_sp = groups.CreateTargetGroup("first group");
  TargetGroupSP second_group_sp = groups.CreateTargetGroup("second group");
  EXPECT_THAT_ERROR(
      first_group_sp->AddTarget(target_sp, TargetGroup::Role::CPU),
      llvm::Succeeded());
  EXPECT_THAT_ERROR(
      second_group_sp->AddTarget(target_sp, TargetGroup::Role::Accelerator),
      llvm::Succeeded());
  EXPECT_THAT_ERROR(
      second_group_sp->AddTarget(survivor_sp, TargetGroup::Role::CPU),
      llvm::Succeeded());

  target_sp->Destroy();

  EXPECT_TRUE(target_sp->GetTargetGroups().empty());
  EXPECT_EQ(std::nullopt, first_group_sp->GetRole(target_sp));
  EXPECT_EQ(std::nullopt, second_group_sp->GetRole(target_sp));
  EXPECT_EQ(std::optional(TargetGroup::Role::CPU),
            second_group_sp->GetRole(survivor_sp));
}

TEST_F(TargetGroupTest, RejectsTargetFromAnotherDebugger) {
  DebuggerSP group_debugger_sp = Debugger::CreateInstance();
  DebuggerSP target_debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(group_debugger_sp);
  ASSERT_TRUE(target_debugger_sp);

  TargetSP target_sp = CreateTarget(*target_debugger_sp);
  ASSERT_TRUE(target_sp);
  TargetGroupSP group_sp =
      group_debugger_sp->GetTargetGroupList().CreateTargetGroup(
          "related targets");
  ASSERT_TRUE(group_sp);

  EXPECT_THAT_ERROR(
      group_sp->AddTarget(target_sp, TargetGroup::Role::CPU),
      llvm::FailedWithMessage("target belongs to a different debugger"));
  EXPECT_TRUE(group_sp->GetTargets().empty());
  EXPECT_TRUE(target_sp->GetTargetGroups().empty());
}

TEST_F(TargetGroupTest, RejectsInvalidTarget) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(*debugger_sp);
  ASSERT_TRUE(target_sp);
  TargetGroupSP group_sp =
      debugger_sp->GetTargetGroupList().CreateTargetGroup("related targets");
  ASSERT_TRUE(group_sp);

  target_sp->Destroy();
  EXPECT_THAT_ERROR(group_sp->AddTarget(target_sp, TargetGroup::Role::CPU),
                    llvm::FailedWithMessage(
                        "cannot add an invalid target to a target group"));
  EXPECT_TRUE(group_sp->GetTargets().empty());
  EXPECT_TRUE(target_sp->GetTargetGroups().empty());
}

TEST_F(TargetGroupTest, ListsNoTargetGroups) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  CommandReturnObject result(/*colors=*/false);
  EXPECT_TRUE(debugger_sp->GetCommandInterpreter().HandleCommand(
      "target groups list", eLazyBoolNo, result));
  EXPECT_TRUE(result.Succeeded());
  EXPECT_TRUE(result.GetErrorString().empty());
  EXPECT_EQ("No target groups.\n", result.GetOutputString());
}

TEST_F(TargetGroupTest, ListsTargetGroupsAndMembers) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP cpu_target_sp = CreateTarget(*debugger_sp);
  TargetSP accelerator_target_sp = CreateTarget(*debugger_sp);
  ASSERT_TRUE(cpu_target_sp);
  ASSERT_TRUE(accelerator_target_sp);

  TargetGroupSP group_sp = debugger_sp->GetTargetGroupList().CreateTargetGroup(
      "CPU and accelerator targets");
  ASSERT_TRUE(group_sp);
  EXPECT_THAT_ERROR(group_sp->AddTarget(cpu_target_sp, TargetGroup::Role::CPU),
                    llvm::Succeeded());
  EXPECT_THAT_ERROR(group_sp->AddTarget(accelerator_target_sp,
                                        TargetGroup::Role::Accelerator),
                    llvm::Succeeded());

  CommandReturnObject result(/*colors=*/false);
  EXPECT_TRUE(debugger_sp->GetCommandInterpreter().HandleCommand(
      "target groups list", eLazyBoolNo, result));
  EXPECT_TRUE(result.Succeeded());
  EXPECT_TRUE(result.GetErrorString().empty());
  EXPECT_EQ("group #0: CPU and accelerator targets\n"
            "  target #0: CPU\n"
            "  target #1: accelerator\n",
            result.GetOutputString());
}

TEST_F(TargetGroupTest, FiltersTargetGroups) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP first_target_sp = CreateTarget(*debugger_sp);
  TargetSP second_target_sp = CreateTarget(*debugger_sp);
  ASSERT_TRUE(first_target_sp);
  ASSERT_TRUE(second_target_sp);

  TargetGroupList &groups = debugger_sp->GetTargetGroupList();
  TargetGroupSP first_group_sp = groups.CreateTargetGroup("first group");
  TargetGroupSP second_group_sp = groups.CreateTargetGroup("second group");
  ASSERT_TRUE(first_group_sp);
  ASSERT_TRUE(second_group_sp);
  EXPECT_THAT_ERROR(
      first_group_sp->AddTarget(first_target_sp, TargetGroup::Role::CPU),
      llvm::Succeeded());
  EXPECT_THAT_ERROR(second_group_sp->AddTarget(second_target_sp,
                                               TargetGroup::Role::Accelerator),
                    llvm::Succeeded());

  {
    CommandReturnObject result(/*colors=*/false);
    EXPECT_TRUE(debugger_sp->GetCommandInterpreter().HandleCommand(
        "target groups list 1", eLazyBoolNo, result));
    EXPECT_TRUE(result.Succeeded());
    EXPECT_TRUE(result.GetErrorString().empty());
    EXPECT_EQ("group #1: second group\n"
              "  target #1: accelerator\n",
              result.GetOutputString());
  }

  {
    CommandReturnObject result(/*colors=*/false);
    EXPECT_TRUE(debugger_sp->GetCommandInterpreter().HandleCommand(
        "target groups list --target 0", eLazyBoolNo, result));
    EXPECT_TRUE(result.Succeeded());
    EXPECT_TRUE(result.GetErrorString().empty());
    EXPECT_EQ("group #0: first group\n"
              "  target #0: CPU\n",
              result.GetOutputString());
  }

  {
    CommandReturnObject result(/*colors=*/false);
    EXPECT_TRUE(debugger_sp->GetCommandInterpreter().HandleCommand(
        "target groups list 1 --target 0", eLazyBoolNo, result));
    EXPECT_TRUE(result.Succeeded());
    EXPECT_TRUE(result.GetErrorString().empty());
    EXPECT_EQ("No matching target groups.\n", result.GetOutputString());
  }
}
