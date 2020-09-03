//===-- ThreadTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Thread.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/Reproducer.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::repro;
using namespace lldb;

namespace {
class ThreadTest : public ::testing::Test {
public:
  void SetUp() override {
    llvm::cantFail(Reproducer::Initialize(ReproducerMode::Off, llvm::None));
    FileSystem::Initialize();
    HostInfo::Initialize();
    platform_linux::PlatformLinux::Initialize();
  }
  void TearDown() override {
    platform_linux::PlatformLinux::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
    Reproducer::Terminate();
  }
};

class DummyProcess : public Process {
public:
  using Process::Process;

  virtual bool CanDebug(lldb::TargetSP target, bool plugin_specified_by_name) {
    return true;
  }
  virtual Status DoDestroy() { return {}; }
  virtual void RefreshStateAfterStop() {}
  virtual size_t DoReadMemory(lldb::addr_t vm_addr, void *buf, size_t size,
                              Status &error) {
    return 0;
  }
  virtual bool UpdateThreadList(ThreadList &old_thread_list,
                                ThreadList &new_thread_list) {
    return false;
  }
  virtual ConstString GetPluginName() { return ConstString("Dummy"); }
  virtual uint32_t GetPluginVersion() { return 0; }

  ProcessModID &GetModIDNonConstRef() { return m_mod_id; }
};

class DummyThread : public Thread {
public:
  using Thread::Thread;

  ~DummyThread() override { DestroyThread(); }

  void RefreshStateAfterStop() override {}

  lldb::RegisterContextSP GetRegisterContext() override { return nullptr; }

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(StackFrame *frame) override {
    return nullptr;
  }

  bool CalculateStopInfo() override { return false; }

  bool IsStillAtLastBreakpointHit() override { return true; }
};
} // namespace

TargetSP CreateTarget(DebuggerSP &debugger_sp, ArchSpec &arch) {
  Status error;
  PlatformSP platform_sp;
  TargetSP target_sp;
  error = debugger_sp->GetTargetList().CreateTarget(
      *debugger_sp, "", arch, eLoadDependentsNo, platform_sp, target_sp);

  if (target_sp) {
    debugger_sp->GetTargetList().SetSelectedTarget(target_sp.get());
  }

  return target_sp;
}

TEST_F(ThreadTest, SetStopInfo) {
  ArchSpec arch("powerpc64-pc-linux");

  Platform::SetHostPlatform(
      platform_linux::PlatformLinux::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  ProcessSP process_sp = std::make_shared<DummyProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);

  DummyProcess *process = static_cast<DummyProcess *>(process_sp.get());

  ThreadSP thread_sp = std::make_shared<DummyThread>(*process_sp.get(), 0);
  ASSERT_TRUE(thread_sp);

  StopInfoSP stopinfo_sp =
      StopInfo::CreateStopReasonWithBreakpointSiteID(*thread_sp.get(), 0);
  ASSERT_TRUE(stopinfo_sp->IsValid() == true);

  /*
   Should make stopinfo valid.
   */
  process->GetModIDNonConstRef().BumpStopID();
  ASSERT_TRUE(stopinfo_sp->IsValid() == false);

  thread_sp->SetStopInfo(stopinfo_sp);
  ASSERT_TRUE(stopinfo_sp->IsValid() == true);
}

TEST_F(ThreadTest, GetPrivateStopInfo) {
  ArchSpec arch("powerpc64-pc-linux");

  Platform::SetHostPlatform(
      platform_linux::PlatformLinux::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  ProcessSP process_sp = std::make_shared<DummyProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);

  DummyProcess *process = static_cast<DummyProcess *>(process_sp.get());

  ThreadSP thread_sp = std::make_shared<DummyThread>(*process_sp.get(), 0);
  ASSERT_TRUE(thread_sp);

  StopInfoSP stopinfo_sp =
      StopInfo::CreateStopReasonWithBreakpointSiteID(*thread_sp.get(), 0);
  ASSERT_TRUE(stopinfo_sp);

  thread_sp->SetStopInfo(stopinfo_sp);

  /*
   Should make stopinfo valid if thread is at last breakpoint hit.
   */
  process->GetModIDNonConstRef().BumpStopID();
  ASSERT_TRUE(stopinfo_sp->IsValid() == false);
  StopInfoSP new_stopinfo_sp = thread_sp->GetPrivateStopInfo();
  ASSERT_TRUE(new_stopinfo_sp && stopinfo_sp->IsValid() == true);
}
