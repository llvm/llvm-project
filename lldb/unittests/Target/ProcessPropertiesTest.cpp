//===----------------------------------------------------------------------===//
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
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ArchSpec.h"
#include "gtest/gtest.h"

#include <mutex>

using namespace lldb_private;
using namespace lldb;

namespace {

// Paths below target.process, as "settings set" would spell them.
constexpr const char *k_reports_all_threads =
    "experimental.os-plugin-reports-all-threads";
constexpr const char *k_trace_thread = "thread.trace-thread";

class DummyProcess : public Process {
public:
  DummyProcess(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {}
  bool CanDebug(lldb::TargetSP, bool) override { return true; }
  size_t DoReadMemory(const ProcessAddress &, void *, size_t,
                      Status &) override {
    return 0;
  }
  Status DoDestroy() override { return {}; }
  void RefreshStateAfterStop() override {}
  bool DoUpdateThreadList(ThreadList &, ThreadList &) override { return false; }
  llvm::StringRef GetPluginName() override { return "Dummy"; }
};

class ProcessPropertiesTest : public ::testing::Test {
public:
  SubsystemRAII<FileSystem, HostInfo, PlatformMacOSX> subsystems;

protected:
  void SetUp() override {
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   [] { Debugger::Initialize(nullptr); });
    Platform::SetHostPlatform(
        PlatformRemoteMacOSX::CreateInstance(true, &m_arch));
    m_debugger_sp = Debugger::CreateInstance();
  }

  void TearDown() override {
    // Both settings are global and outlive this test.
    Process::GetGlobalProperties().SetPropertyValue(
        nullptr, eVarSetOperationClear, k_reports_all_threads, "");
    Process::GetGlobalProperties().SetPropertyValue(
        nullptr, eVarSetOperationClear, k_trace_thread, "");
    if (m_debugger_sp)
      Debugger::Destroy(m_debugger_sp);
  }

  ProcessSP CreateProcess() {
    PlatformSP platform_sp;
    TargetSP target_sp;
    m_debugger_sp->GetTargetList().CreateTarget(
        *m_debugger_sp, "", m_arch, eLoadDependentsNo, platform_sp, target_sp);
    if (!target_sp)
      return nullptr;
    return std::make_shared<DummyProcess>(target_sp,
                                          Listener::MakeListener("dummy"));
  }

  static void SetReportsAllThreads(const char *value) {
    Process::GetGlobalProperties().SetPropertyValue(
        nullptr, eVarSetOperationAssign, k_reports_all_threads, value);
  }

  ArchSpec m_arch = ArchSpec("x86_64-apple-macosx-");
  DebuggerSP m_debugger_sp;
};

} // namespace

TEST_F(ProcessPropertiesTest, OSPluginReportsAllThreadsDefaultsToTrue) {
  ASSERT_TRUE(m_debugger_sp);
  EXPECT_TRUE(Process::GetGlobalProperties().GetOSPluginReportsAllThreads());

  ProcessSP process_sp = CreateProcess();
  ASSERT_TRUE(process_sp);
  EXPECT_TRUE(process_sp->GetOSPluginReportsAllThreads());
}

// "settings set target.process.experimental.os-plugin-reports-all-threads"
// must reach what a process reads, not only the global properties.
TEST_F(ProcessPropertiesTest, OSPluginReportsAllThreadsFollowsGlobalSetting) {
  ASSERT_TRUE(m_debugger_sp);
  ProcessSP process_sp = CreateProcess();
  ASSERT_TRUE(process_sp);

  SetReportsAllThreads("false");
  EXPECT_FALSE(Process::GetGlobalProperties().GetOSPluginReportsAllThreads());
  EXPECT_FALSE(process_sp->GetOSPluginReportsAllThreads());

  // The collection is shared, so a process created afterwards sees it too.
  ProcessSP later_process_sp = CreateProcess();
  ASSERT_TRUE(later_process_sp);
  EXPECT_FALSE(later_process_sp->GetOSPluginReportsAllThreads());

  SetReportsAllThreads("true");
  EXPECT_TRUE(Process::GetGlobalProperties().GetOSPluginReportsAllThreads());
  EXPECT_TRUE(process_sp->GetOSPluginReportsAllThreads());
}

// The setter must write the experimental collection and leave the thread
// collection alone.
TEST_F(ProcessPropertiesTest, SetOSPluginReportsAllThreadsWritesTheSetting) {
  ASSERT_TRUE(m_debugger_sp);
  ProcessSP process_sp = CreateProcess();
  ASSERT_TRUE(process_sp);

  const bool trace_thread =
      Thread::GetGlobalProperties().GetTraceEnabledState();

  process_sp->SetOSPluginReportsAllThreads(false);
  EXPECT_FALSE(process_sp->GetOSPluginReportsAllThreads());
  EXPECT_EQ(trace_thread, Thread::GetGlobalProperties().GetTraceEnabledState());

  process_sp->SetOSPluginReportsAllThreads(true);
  EXPECT_TRUE(process_sp->GetOSPluginReportsAllThreads());
  EXPECT_EQ(trace_thread, Thread::GetGlobalProperties().GetTraceEnabledState());

  // The collection is shared, so writing through a process is visible on the
  // global properties.
  EXPECT_TRUE(Process::GetGlobalProperties().GetOSPluginReportsAllThreads());
}
