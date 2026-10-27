//===-- ThreadPlanStackTest.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStack.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadList.h"
#include "lldb/Utility/Listener.h"
#include "gtest/gtest.h"

#include <limits>
#include <mutex>

using namespace lldb;
using namespace lldb_private;

namespace {
class TestProcess : public Process {
public:
  TestProcess(TargetSP target, ListenerSP listener)
      : Process(target, listener) {}

  bool CanDebug(TargetSP, bool) override { return true; }
  Status DoDestroy() override { return {}; }
  void RefreshStateAfterStop() override {}
  size_t DoReadMemory(const ProcessAddress &, void *, size_t,
                      Status &) override {
    return 0;
  }
  bool DoUpdateThreadList(ThreadList &, ThreadList &new_threads) override {
    if (!refresh_threads)
      return false;
    ++refresh_count;
    for (const auto &thread : next_threads)
      new_threads.AddThread(thread);
    return true;
  }
  llvm::StringRef GetPluginName() override { return "thread-plan-test"; }

  ThreadPlanStackMap &Plans() { return m_thread_plans; }
  void PrepareThreadRefresh(std::vector<ThreadSP> threads) {
    next_threads = std::move(threads);
    refresh_threads = true;
    m_current_private_state_thread_sp->SetPrivateState(eStateStopped);
    m_mod_id.BumpStopID();
  }

  unsigned refresh_count = 0;

private:
  bool refresh_threads = false;
  std::vector<ThreadSP> next_threads;
};

class TestThread : public Thread {
public:
  using Thread::Thread;
  ~TestThread() override { DestroyThread(); }
  void RefreshStateAfterStop() override {}
  RegisterContextSP GetRegisterContext() override { return nullptr; }
  RegisterContextSP CreateRegisterContextForFrame(StackFrame *) override {
    return nullptr;
  }
  bool CalculateStopInfo() override { return false; }
};

class ThreadPlanStackMapTest : public testing::Test {
protected:
  static void SetUpTestSuite() {
    FileSystem::Initialize();
    HostInfo::Initialize();
    platform_linux::PlatformLinux::Initialize();
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   [] { Debugger::Initialize(nullptr); });
  }
  static void TearDownTestSuite() {
    platform_linux::PlatformLinux::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
  void SetUp() override {
    debugger = Debugger::CreateInstance();
    ASSERT_TRUE(debugger);
    ArchSpec arch(HostInfo::GetTargetTriple());
    PlatformSP platform =
        platform_linux::PlatformLinux::CreateInstance(true, &arch);
    ASSERT_TRUE(platform);
    Platform::SetHostPlatform(platform);
    TargetSP target;
    ASSERT_TRUE(debugger->GetTargetList()
                    .CreateTarget(*debugger, "", arch, eLoadDependentsNo,
                                  platform, target)
                    .Success());
    process = std::make_shared<TestProcess>(
        target, Listener::MakeListener("thread-plan-test"));
  }

  ThreadSP AddThread(tid_t tid) {
    auto thread = std::make_shared<TestThread>(*process, tid);
    process->GetThreadList().AddThread(thread);
    return thread;
  }
  void Update(bool delete_missing = true, bool check_for_new = true) {
    process->Plans().Update(process->GetThreadList(), delete_missing,
                            check_for_new);
  }

  DebuggerSP debugger;
  std::shared_ptr<TestProcess> process;
};
} // namespace

TEST_F(ThreadPlanStackMapTest, RemovesOnlyMissingPlans) {
  const tid_t live_tid = std::numeric_limits<tid_t>::max() - 1;
  AddThread(live_tid);
  auto removed = AddThread(42);
  Update();
  auto *survivor = process->Plans().Find(live_tid);
  ASSERT_NE(nullptr, survivor);
  ASSERT_NE(nullptr, process->Plans().Find(42));
  process->GetThreadList().RemoveThreadByID(42, false);
  Update();
  EXPECT_EQ(survivor, process->Plans().Find(live_tid));
  EXPECT_EQ(nullptr, process->Plans().Find(42));
}

TEST_F(ThreadPlanStackMapTest, PreservesMissingPlansWhenRequested) {
  auto removed = AddThread(42);
  Update();
  auto *plans = process->Plans().Find(42);
  ASSERT_NE(nullptr, plans);
  process->GetThreadList().Clear();
  Update(false, false);
  EXPECT_EQ(plans, process->Plans().Find(42));
}

TEST_F(ThreadPlanStackMapTest, RemovesAllPlansForAnEmptyList) {
  auto first = AddThread(7);
  auto second = AddThread(42);
  Update();
  ASSERT_NE(nullptr, process->Plans().Find(7));
  ASSERT_NE(nullptr, process->Plans().Find(42));
  process->GetThreadList().Clear();
  Update(true, false);
  EXPECT_EQ(nullptr, process->Plans().Find(7));
  EXPECT_EQ(nullptr, process->Plans().Find(42));
}

TEST_F(ThreadPlanStackMapTest, DoesNotRefreshProcessWhilePruningSnapshot) {
  auto survivor = AddThread(7);
  auto removed = AddThread(42);
  Update();
  ASSERT_NE(nullptr, process->Plans().Find(7));
  ASSERT_NE(nullptr, process->Plans().Find(42));
  process->PrepareThreadRefresh({survivor});

  // The caller supplies a current snapshot. Pruning must not refresh the
  // process's separate, stale list as a side effect of inspecting this one.
  ThreadList snapshot(*process);
  snapshot.AddThread(survivor);
  snapshot.SetStopID(process->GetStopID());
  process->Plans().Update(snapshot, true, false);

  EXPECT_EQ(0u, process->refresh_count);
  EXPECT_NE(nullptr, process->Plans().Find(7));
  EXPECT_EQ(nullptr, process->Plans().Find(42));

  process->UpdateThreadListIfNeeded();
  EXPECT_EQ(1u, process->refresh_count);
}
