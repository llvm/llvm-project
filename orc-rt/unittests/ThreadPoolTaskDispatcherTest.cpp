//===-- ThreadPoolTaskDispatcherTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt/ThreadPoolTaskDispatcher.h"
#include "gtest/gtest.h"

#include <atomic>
#include <future>
#include <thread>
#include <vector>

using namespace orc_rt;

namespace {

TEST(ThreadPoolTaskDispatcherTest, NoTasks) {
  // Check that immediate shutdown works as expected.
  ThreadPoolTaskDispatcher Dispatcher(1);
  Dispatcher.shutdown();
}

TEST(ThreadPoolTaskDispatcherTest, BasicTaskExecution) {
  // Smoke test: Check that we can run a single task on a single-threaded pool.
  ThreadPoolTaskDispatcher Dispatcher(1);
  std::atomic<bool> TaskRan = false;

  Dispatcher.dispatch(makeGenericTask([&]() { TaskRan = true; }));

  Dispatcher.shutdown();

  EXPECT_TRUE(TaskRan);
}

TEST(ThreadPoolTaskDispatcherTest, SingleThreadMultipleTasks) {
  // Check that multiple tasks in a single threaded pool run as expected.
  ThreadPoolTaskDispatcher Dispatcher(1);
  size_t NumTasksToRun = 10;
  std::atomic<size_t> TasksRun = 0;

  for (size_t I = 0; I != NumTasksToRun; ++I)
    Dispatcher.dispatch(makeGenericTask([&]() { ++TasksRun; }));

  Dispatcher.shutdown();

  EXPECT_EQ(TasksRun, NumTasksToRun);
}

TEST(ThreadPoolTaskDispatcherTest, ConcurrentTasks) {
  // Check that tasks are run concurrently when multiple workers are available.
  // Adds two tasks that communicate a value back and forth using futures.
  // Neither task should be able to complete without the other having started.
  ThreadPoolTaskDispatcher Dispatcher(2);

  std::promise<int> PInit;
  std::future<int> FInit = PInit.get_future();
  std::promise<int> P1;
  std::future<int> F1 = P1.get_future();
  std::promise<int> P2;
  std::future<int> F2 = P2.get_future();
  std::promise<int> PResult;
  std::future<int> FResult = PResult.get_future();

  // Task A gets the initial value, sends it via P1, waits for response on F2.
  Dispatcher.dispatch(makeGenericTask([&]() {
    P1.set_value(FInit.get());
    PResult.set_value(F2.get());
  }));

  // Task B gets value from F1, sends it back on P2.
  Dispatcher.dispatch(makeGenericTask([&]() { P2.set_value(F1.get()); }));

  int ExpectedValue = 42;
  PInit.set_value(ExpectedValue);

  Dispatcher.shutdown();

  EXPECT_EQ(FResult.get(), ExpectedValue);
}

TEST(ThreadPoolTaskDispatcherTest, TasksRejectedAfterShutdown) {
  class TaskToReject : public Task {
  public:
    TaskToReject(bool &BodyRun, bool &DestructorRun)
        : BodyRun(BodyRun), DestructorRun(DestructorRun) {}
    ~TaskToReject() { DestructorRun = true; }
    void run() override { BodyRun = true; }

  private:
    bool &BodyRun;
    bool &DestructorRun;
  };

  ThreadPoolTaskDispatcher Dispatcher(1);
  Dispatcher.shutdown();

  bool BodyRun = false;
  bool DestructorRun = false;

  Dispatcher.dispatch(std::make_unique<TaskToReject>(BodyRun, DestructorRun));

  EXPECT_FALSE(BodyRun);
  EXPECT_TRUE(DestructorRun);
}

} // end anonymous namespace
