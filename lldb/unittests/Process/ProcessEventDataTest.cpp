//===-- ProcessEventDataTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/Event.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::repro;
using namespace lldb;

namespace {
class ProcessEventDataTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
    PlatformMacOSX::Initialize();
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
  }
  void TearDown() override {
    PlatformMacOSX::Terminate();
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
};

class DummyStopInfo : public StopInfo {
public:
  DummyStopInfo(Thread &thread, uint64_t value) : StopInfo(thread, value) {}

  bool ShouldStop(Event *event_ptr) override { return m_should_stop; }

  StopReason GetStopReason() const override { return m_stop_reason; }

  bool m_should_stop = true;
  StopReason m_stop_reason = eStopReasonBreakpoint;
};

class DummyProcessEventData : public Process::ProcessEventData {
public:
  DummyProcessEventData(ProcessSP &process_sp, StateType state)
      : ProcessEventData(process_sp, state) {}
  bool ShouldStop(Event *event_ptr, bool &found_valid_stopinfo) override {
    m_should_stop_hit_count++;
    return false;
  }

  int m_should_stop_hit_count = 0;
};
} // namespace

typedef std::shared_ptr<Process::ProcessEventData> ProcessEventDataSP;
typedef std::shared_ptr<Event> EventSP;

TargetSP CreateTarget(DebuggerSP &debugger_sp, ArchSpec &arch) {
  PlatformSP platform_sp;
  TargetSP target_sp;
  debugger_sp->GetTargetList().CreateTarget(
      *debugger_sp, "", arch, eLoadDependentsNo, platform_sp, target_sp);

  return target_sp;
}

ThreadSP CreateThread(ProcessSP &process_sp, bool should_stop,
                      bool has_valid_stopinfo) {
  ThreadSP thread_sp = std::make_shared<DummyThread>(*process_sp.get(), 0);
  if (thread_sp == nullptr) {
    return nullptr;
  }

  if (has_valid_stopinfo) {
    StopInfoSP stopinfo_sp =
        std::make_shared<DummyStopInfo>(*thread_sp.get(), 0);
    static_cast<DummyStopInfo *>(stopinfo_sp.get())->m_should_stop =
        should_stop;
    if (stopinfo_sp == nullptr) {
      return nullptr;
    }

    thread_sp->SetStopInfo(stopinfo_sp);
  }

  process_sp->GetThreadList().AddThread(thread_sp);

  return thread_sp;
}

TEST_F(ProcessEventDataTest, DoOnRemoval) {
  ArchSpec arch("x86_64-apple-macosx-");

  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  ProcessSP process_sp = std::make_shared<DummyProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);

  /*
   Should hit ShouldStop if state is eStateStopped
   */
  ProcessEventDataSP event_data_sp =
      std::make_shared<DummyProcessEventData>(process_sp, eStateStopped);
  EventSP event_sp = std::make_shared<Event>(0, event_data_sp);
  event_data_sp->SetUpdateStateOnRemoval(event_sp.get());
  event_data_sp->DoOnRemoval(event_sp.get());
  bool result = static_cast<DummyProcessEventData *>(event_data_sp.get())
                    ->m_should_stop_hit_count == 1;
  ASSERT_TRUE(result);

  /*
   Should not hit ShouldStop if state is not eStateStopped
   */
  event_data_sp =
      std::make_shared<DummyProcessEventData>(process_sp, eStateStepping);
  event_sp = std::make_shared<Event>(0, event_data_sp);
  event_data_sp->SetUpdateStateOnRemoval(event_sp.get());
  event_data_sp->DoOnRemoval(event_sp.get());
  result = static_cast<DummyProcessEventData *>(event_data_sp.get())
               ->m_should_stop_hit_count == 0;
  ASSERT_TRUE(result);
}

TEST_F(ProcessEventDataTest, ShouldStop) {
  ArchSpec arch("x86_64-apple-macosx-");

  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  ProcessSP process_sp = std::make_shared<DummyProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);

  // wants to stop and has valid StopInfo
  ThreadSP thread_sp = CreateThread(process_sp, true, true);

  ProcessEventDataSP event_data_sp =
      std::make_shared<Process::ProcessEventData>(process_sp, eStateStopped);
  EventSP event_sp = std::make_shared<Event>(0, event_data_sp);
  /*
   Should stop if thread has valid StopInfo and not suspended
   */
  bool found_valid_stopinfo = false;
  bool should_stop =
      event_data_sp->ShouldStop(event_sp.get(), found_valid_stopinfo);
  ASSERT_TRUE(should_stop == true && found_valid_stopinfo == true);

  /*
   Should not stop if thread has valid StopInfo but was suspended
   */
  found_valid_stopinfo = false;
  thread_sp->SetResumeState(eStateSuspended);
  should_stop = event_data_sp->ShouldStop(event_sp.get(), found_valid_stopinfo);
  ASSERT_TRUE(should_stop == false && found_valid_stopinfo == false);

  /*
   Should not stop, thread-reason of stop does not want to stop in its
   callback and suspended thread who wants to (from previous stop)
   must be ignored.
   */
  event_data_sp =
      std::make_shared<Process::ProcessEventData>(process_sp, eStateStopped);
  event_sp = std::make_shared<Event>(0, event_data_sp);
  process_sp->GetThreadList().Clear();

  for (int i = 0; i < 6; i++) {
    if (i == 2) {
      // Does not want to stop but has valid StopInfo
      thread_sp = CreateThread(process_sp, false, true);
    } else if (i == 5) {
      // Wants to stop and has valid StopInfo
      thread_sp = CreateThread(process_sp, true, true);
      thread_sp->SetResumeState(eStateSuspended);
    } else {
      // Thread has no StopInfo i.e is not the reason of stop
      thread_sp = CreateThread(process_sp, false, false);
    }
  }
  ASSERT_TRUE(process_sp->GetThreadList().GetSize() == 6);

  found_valid_stopinfo = false;
  should_stop = event_data_sp->ShouldStop(event_sp.get(), found_valid_stopinfo);
  ASSERT_TRUE(should_stop == false && found_valid_stopinfo == true);
}
