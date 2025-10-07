//===-- BreakpointClearConditionTest.cpp
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
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBBreakpointLocation.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBTarget.h"
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

class BreakpointClearConditionTest : public ::testing::Test {
public:
  void SetUp() override {
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { SBDebugger::Initialize(); });
  };

  SBDebugger m_sb_debugger;
  SubsystemRAII<FileSystem, HostInfo, PlatformMacOSX> subsystems;
};

template<typename T>
void test_condition(T sb_object) {
  const char *in_cond_str = "Here is a condition";
  sb_object.SetCondition(in_cond_str);
  // Make sure we set the condition correctly:
  const char *out_cond_str = sb_object.GetCondition();
  EXPECT_STREQ(in_cond_str, out_cond_str);
  // Now unset it by passing in nullptr and make sure that works:
  const char *empty_tokens[2] = {nullptr, ""};
  for (auto token : empty_tokens) {
    sb_object.SetCondition(token);
    out_cond_str = sb_object.GetCondition();
    // And make sure an unset condition returns nullptr:
    EXPECT_EQ(nullptr, out_cond_str);
  }  
}

TEST_F(BreakpointClearConditionTest, BreakpointClearConditionTest) {
  // Set up the debugger, make sure that was done properly.
  m_sb_debugger = SBDebugger::Create(false);

  // Create target
  SBTarget sb_target;
  SBError error;
  sb_target = m_sb_debugger.CreateTarget("", "x86_64-apple-macosx-", "remote-macosx",
       /*add_dependent=*/ false, error);

  EXPECT_EQ(sb_target.IsValid(), true);

  // Create breakpoint
  SBBreakpoint sb_breakpoint =
      sb_target.BreakpointCreateByAddress(0xDEADBEEF);
  test_condition(sb_breakpoint);
  
  // Address breakpoints always have one location, so we can also use this
  // to test the location:
  SBBreakpointLocation sb_loc = sb_breakpoint.GetLocationAtIndex(0);
  EXPECT_EQ(sb_loc.IsValid(), true);
  test_condition(sb_loc);
  
}
