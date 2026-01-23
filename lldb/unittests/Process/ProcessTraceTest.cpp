//===-- ProcessEventDataTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ProcessTrace.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/HostInfo.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;
using namespace platform_linux;

// This is needed for the tests that create a trace process.
class ProcessTraceTest : public ::testing::Test {
public:
  void SetUp() override {
    ProcessTrace::Initialize();
    FileSystem::Initialize();
    HostInfo::Initialize();
    PlatformLinux::Initialize();
  }
  void TearDown() override {
    PlatformLinux::Initialize();
    HostInfo::Terminate();
    FileSystem::Terminate();
    ProcessTrace::Terminate();
  }
};

TargetSP CreateTarget(DebuggerSP &debugger_sp, const ArchSpec &arch) {
  PlatformSP platform_sp;
  TargetSP target_sp;
  debugger_sp->GetTargetList().CreateTarget(
      *debugger_sp, "", arch, eLoadDependentsNo, platform_sp, target_sp);
  return target_sp;
}

// Test that we can create a process trace with a nullptr core file.
TEST_F(ProcessTraceTest, ConstructorWithNullptrCoreFile) {
  ArchSpec arch("i386-pc-linux");

  Platform::SetHostPlatform(PlatformLinux::CreateInstance(true, &arch));
  ASSERT_NE(Platform::GetHostPlatform(), nullptr);

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  ProcessSP process_sp = target_sp->CreateProcess(
      /*listener*/ nullptr, "trace",
      /*crash_file*/ nullptr,
      /*can_connect*/ false);

  ASSERT_NE(process_sp, nullptr);
}
