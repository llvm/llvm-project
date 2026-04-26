//===-- DebuggerTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Debugger.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/ExecutionContext.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

namespace {
class DebuggerTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
    PlatformMacOSX::Initialize();
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
    ArchSpec arch("x86_64-apple-macosx-");
    Platform::SetHostPlatform(
        PlatformRemoteMacOSX::CreateInstance(true, &arch));
  }
  void TearDown() override {
    PlatformMacOSX::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};
} // namespace

TEST_F(DebuggerTest, TestSettings) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();

  EXPECT_TRUE(debugger_sp->SetUseColor(true));
  EXPECT_TRUE(debugger_sp->GetUseColor());

  FormatEntity::Entry format("foo");
  EXPECT_TRUE(debugger_sp->SetStatuslineFormat(format));
  EXPECT_EQ(debugger_sp->GetStatuslineFormat().string, "foo");

  Debugger::Destroy(debugger_sp);
}

TEST_F(DebuggerTest,
       SelectedExecutionContextUsesDummyTargetWhenNoTargetSelected) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();

  // No targets have been added, so no target is selected.
  ASSERT_EQ(debugger_sp->GetSelectedTarget().get(), nullptr);

  Target &dummy_target = debugger_sp->GetDummyTarget();

  {
    ExecutionContextRef exe_ctx_ref =
        debugger_sp->GetSelectedExecutionContextRef(
            /*adopt_dummy_target=*/true);
    EXPECT_EQ(exe_ctx_ref.GetTargetSP().get(), &dummy_target);

    ExecutionContext exe_ctx =
        debugger_sp->GetSelectedExecutionContext(/*adopt_dummy_target=*/true);
    EXPECT_EQ(exe_ctx.GetTargetPtr(), &dummy_target);
  }

  {
    ExecutionContextRef exe_ctx_ref =
        debugger_sp->GetSelectedExecutionContextRef(
            /*adopt_dummy_target=*/false);
    EXPECT_EQ(exe_ctx_ref.GetTargetSP().get(), nullptr);

    ExecutionContext exe_ctx =
        debugger_sp->GetSelectedExecutionContext(/*adopt_dummy_target=*/false);
    EXPECT_EQ(exe_ctx.GetTargetPtr(), nullptr);
  }

  Debugger::Destroy(debugger_sp);
}
