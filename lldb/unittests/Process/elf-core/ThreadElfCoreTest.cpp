//===-- ThreadElfCoreTest.cpp ------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Plugins/Process/elf-core/ThreadElfCore.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/Listener.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

#include <memory>
#include <mutex>
#include <sys/resource.h>
#include <unistd.h>

#ifndef HAVE_GETTID
#include <sys/syscall.h>
pid_t gettid() { return ((pid_t)syscall(SYS_gettid)); }
#endif

using namespace lldb_private;

namespace {

struct ElfCoreTest : public testing::Test {
  static void SetUpTestCase() {
    FileSystem::Initialize();
    HostInfo::Initialize();
    platform_linux::PlatformLinux::Initialize();
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
  }
  static void TearDownTestCase() {
    platform_linux::PlatformLinux::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};

struct DummyProcess : public Process {
  DummyProcess(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {
    SetID(getpid());
  }

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
};

struct DummyThread : public Thread {
  using Thread::Thread;

  ~DummyThread() override { DestroyThread(); }

  void RefreshStateAfterStop() override {}

  lldb::RegisterContextSP GetRegisterContext() override { return nullptr; }

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(StackFrame *frame) override {
    return nullptr;
  }

  bool CalculateStopInfo() override { return false; }
};

lldb::TargetSP CreateTarget(lldb::DebuggerSP &debugger_sp, ArchSpec &arch) {
  lldb::PlatformSP platform_sp;
  lldb::TargetSP target_sp;
  debugger_sp->GetTargetList().CreateTarget(
      *debugger_sp, "", arch, eLoadDependentsNo, platform_sp, target_sp);
  return target_sp;
}

lldb::ThreadSP CreateThread(lldb::ProcessSP &process_sp) {
  lldb::ThreadSP thread_sp =
      std::make_shared<DummyThread>(*process_sp.get(), gettid());
  if (thread_sp == nullptr) {
    return nullptr;
  }
  process_sp->GetThreadList().AddThread(thread_sp);

  return thread_sp;
}

} // namespace

TEST_F(ElfCoreTest, PopulatePrpsInfoTest) {
  ArchSpec arch{HostInfo::GetTargetTriple()};
  lldb::DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  lldb::TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  lldb::ListenerSP listener_sp(Listener::MakeListener("dummy"));
  lldb::ProcessSP process_sp =
      std::make_shared<DummyProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);
  auto prpsinfo_opt = ELFLinuxPrPsInfo::Populate(process_sp);
  ASSERT_TRUE(prpsinfo_opt.has_value());
  ASSERT_EQ(prpsinfo_opt->pr_pid, getpid());
  ASSERT_EQ(prpsinfo_opt->pr_state, 0);
  ASSERT_EQ(prpsinfo_opt->pr_sname, 'R');
  ASSERT_EQ(prpsinfo_opt->pr_zomb, 0);
  int priority = getpriority(PRIO_PROCESS, getpid());
  if (priority == -1)
    ASSERT_EQ(errno, 0);
  ASSERT_EQ(prpsinfo_opt->pr_nice, priority);
  ASSERT_EQ(prpsinfo_opt->pr_flag, 0UL);
  ASSERT_EQ(prpsinfo_opt->pr_uid, getuid());
  ASSERT_EQ(prpsinfo_opt->pr_gid, getgid());
  ASSERT_EQ(prpsinfo_opt->pr_pid, getpid());
  ASSERT_EQ(prpsinfo_opt->pr_ppid, getppid());
  ASSERT_EQ(prpsinfo_opt->pr_pgrp, getpgrp());
  ASSERT_EQ(prpsinfo_opt->pr_sid, getsid(getpid()));
  ASSERT_EQ(std::string{prpsinfo_opt->pr_fname}, "ProcessElfCoreT");
  ASSERT_TRUE(std::string{prpsinfo_opt->pr_psargs}.empty());
  lldb_private::ProcessInstanceInfo info;
  ASSERT_TRUE(process_sp->GetProcessInfo(info));
  const char *args[] = {"a.out", "--foo=bar", "--baz=boo", nullptr};
  info.SetArguments(args, true);
  prpsinfo_opt =
      ELFLinuxPrPsInfo::Populate(info, lldb::StateType::eStateStopped);
  ASSERT_TRUE(prpsinfo_opt.has_value());
  ASSERT_EQ(prpsinfo_opt->pr_pid, getpid());
  ASSERT_EQ(prpsinfo_opt->pr_state, 3);
  ASSERT_EQ(prpsinfo_opt->pr_sname, 'T');
  ASSERT_EQ(std::string{prpsinfo_opt->pr_fname}, "a.out");
  ASSERT_FALSE(std::string{prpsinfo_opt->pr_psargs}.empty());
  ASSERT_EQ(std::string{prpsinfo_opt->pr_psargs}, "a.out --foo=bar --baz=boo");
}

TEST_F(ElfCoreTest, PopulatePrStatusTest) {
  ArchSpec arch{HostInfo::GetTargetTriple()};
  lldb::DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  lldb::TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  lldb::ListenerSP listener_sp(Listener::MakeListener("dummy"));
  lldb::ProcessSP process_sp =
      std::make_shared<DummyProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);
  lldb::ThreadSP thread_sp = CreateThread(process_sp);
  ASSERT_TRUE(thread_sp);
  auto prstatus_opt = ELFLinuxPrStatus::Populate(thread_sp);
  ASSERT_TRUE(prstatus_opt.has_value());
  ASSERT_EQ(prstatus_opt->si_signo, 0);
  ASSERT_EQ(prstatus_opt->si_code, 0);
  ASSERT_EQ(prstatus_opt->si_errno, 0);
  ASSERT_EQ(prstatus_opt->pr_cursig, 0);
  ASSERT_EQ(prstatus_opt->pr_sigpend, 0UL);
  ASSERT_EQ(prstatus_opt->pr_sighold, 0UL);
  ASSERT_EQ(prstatus_opt->pr_pid, static_cast<uint32_t>(gettid()));
  ASSERT_EQ(prstatus_opt->pr_ppid, static_cast<uint32_t>(getppid()));
  ASSERT_EQ(prstatus_opt->pr_pgrp, static_cast<uint32_t>(getpgrp()));
  ASSERT_EQ(prstatus_opt->pr_sid, static_cast<uint32_t>(getsid(gettid())));
}
