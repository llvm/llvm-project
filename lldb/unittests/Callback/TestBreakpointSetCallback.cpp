//===-- TestBreakpointSetCallback.cpp
//--------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Progress.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/lldb-private-enumerations.h"
#include "lldb/lldb-types.h"
#include "gtest/gtest.h"
#include <iostream>
#include <memory>
#include <mutex>

using namespace lldb_private;
using namespace lldb;

static constexpr lldb::user_id_t expected_breakpoint_id = 1;
static constexpr lldb::user_id_t expected_breakpoint_location_id = 0;

int baton_value;

class BreakpointSetCallbackTest : public ::testing::Test {
public:
  static void CheckCallbackArgs(void *baton, StoppointCallbackContext *context,
                                lldb::user_id_t break_id,
                                lldb::user_id_t break_loc_id,
                                TargetSP expected_target_sp) {
    EXPECT_EQ(context->exe_ctx_ref.GetTargetSP(), expected_target_sp);
    EXPECT_EQ(baton, &baton_value);
    EXPECT_EQ(break_id, expected_breakpoint_id);
    EXPECT_EQ(break_loc_id, expected_breakpoint_location_id);
  }

protected:
  void SetUp() override {
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
  };

  DebuggerSP m_debugger_sp;
  PlatformSP m_platform_sp;
  SubsystemRAII<FileSystem, HostInfo, PlatformMacOSX, ProgressManager>
      subsystems;
};

TEST_F(BreakpointSetCallbackTest, TestBreakpointSetCallback) {
  // Set up the debugger, make sure that was done properly.
  TargetSP target_sp;
  ArchSpec arch("x86_64-apple-macosx-");
  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  m_debugger_sp = Debugger::CreateInstance();

  // Create target
  m_debugger_sp->GetTargetList().CreateTarget(*m_debugger_sp, "", arch,
                                              lldb_private::eLoadDependentsNo,
                                              m_platform_sp, target_sp);

  // Create breakpoint
  BreakpointSP breakpoint_sp =
      target_sp->CreateBreakpoint(0xDEADBEEF, false, false);

  breakpoint_sp->SetCallback(
      [target_sp](void *baton, StoppointCallbackContext *context,
                  lldb::user_id_t break_id, lldb::user_id_t break_loc_id) {
        CheckCallbackArgs(baton, context, break_id, break_loc_id, target_sp);
        return true;
      },
      (void *)&baton_value, true);
  ExecutionContext exe_ctx(target_sp, false);
  StoppointCallbackContext context(nullptr, exe_ctx, true);
  breakpoint_sp->InvokeCallback(&context, 0);
}
