//===-- ThreadTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Thread.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include <thread>
#ifdef _WIN32
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"

#include "Plugins/Platform/Windows/PlatformWindows.h"
#include "Plugins/Process/Windows/Common/LocalDebugDelegate.h"
#include "Plugins/Process/Windows/Common/ProcessWindows.h"
#include "Plugins/Process/Windows/Common/TargetThreadWindows.h"
#endif
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Utility/ArchSpec.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::repro;
using namespace lldb;

namespace {

#ifdef _WIN32
using SetThreadDescriptionFunctionPtr =
    HRESULT(WINAPI *)(HANDLE hThread, PCWSTR lpThreadDescription);

static SetThreadDescriptionFunctionPtr SetThreadName;
#endif

class ThreadTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
#ifdef _WIN32
    HMODULE hModule = ::LoadLibraryW(L"Kernel32.dll");
    if (hModule) {
      SetThreadName = reinterpret_cast<SetThreadDescriptionFunctionPtr>(
          ::GetProcAddress(hModule, "SetThreadDescription"));
    }
    PlatformWindows::Initialize();
#endif
    platform_linux::PlatformLinux::Initialize();
  }
  void TearDown() override {
#ifdef _WIN32
    PlatformWindows::Terminate();
#endif
    platform_linux::PlatformLinux::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};

class DummyProcess : public Process {
public:
  DummyProcess(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {}

  bool CanDebug(lldb::TargetSP target, bool plugin_specified_by_name) override {
    return true;
  }
  Status DoDestroy() override { return {}; }
  void RefreshStateAfterStop() override {}
  size_t DoReadMemory(lldb::addr_t vm_addr, void *buf, size_t size,
                      Status &error) override {
    return 0;
  }
  bool DoUpdateThreadList(ThreadList &old_thread_list,
                          ThreadList &new_thread_list) override {
    return false;
  }
  llvm::StringRef GetPluginName() override { return "Dummy"; }

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
  PlatformSP platform_sp;
  TargetSP target_sp;
  debugger_sp->GetTargetList().CreateTarget(
      *debugger_sp, "", arch, eLoadDependentsNo, platform_sp, target_sp);

  return target_sp;
}

#ifdef _WIN32
std::shared_ptr<TargetThreadWindows>
CreateWindowsThread(const ProcessWindowsSP &process_sp, std::thread &t) {
  HostThread host_thread((lldb::thread_t)t.native_handle());
  ThreadSP thread_sp =
      std::make_shared<TargetThreadWindows>(*process_sp.get(), host_thread);
  return std::static_pointer_cast<TargetThreadWindows>(thread_sp);
}

TEST_F(ThreadTest, GetThreadDescription) {
  if (!SetThreadName)
    return;

  ArchSpec arch(HostInfo::GetArchitecture());
  Platform::SetHostPlatform(PlatformWindows::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  auto process_sp = std::static_pointer_cast<ProcessWindows>(
      ProcessWindows::CreateInstance(target_sp, listener_sp, nullptr, false));
  ASSERT_TRUE(process_sp);

  std::thread t([]() {});
  auto thread_sp = CreateWindowsThread(process_sp, t);
  DWORD tid = thread_sp->GetHostThread().GetNativeThread().GetThreadId();
  HANDLE hThread = ::OpenThread(THREAD_SET_LIMITED_INFORMATION, FALSE, tid);
  ASSERT_TRUE(hThread);

  SetThreadName(hThread, L"thread name");
  ::CloseHandle(hThread);
  ASSERT_STREQ(thread_sp->GetName(), "thread name");

  t.join();
}
#endif

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
