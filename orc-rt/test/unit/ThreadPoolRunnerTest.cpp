//===- ThreadPoolRunnerTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt/ThreadPoolRunner.h"
#include "gtest/gtest.h"

#include <atomic>
#include <future>

using namespace orc_rt;

namespace {

TEST(ThreadPoolRunnerTest, NoTasks) {
  // Check that immediate destruction works as expected.
  ThreadPoolRunner R(1);
}

TEST(ThreadPoolRunnerTest, BasicTaskExecution) {
  // Smoke test: dispatch one task on a single-threaded pool, wait for it to
  // run, then let the runner destruct.
  std::promise<void> Done;
  std::future<void> DoneF = Done.get_future();

  {
    ThreadPoolRunner R(1);
    R([&]() { Done.set_value(); });
    DoneF.get();
  }
}

TEST(ThreadPoolRunnerTest, SingleThreadMultipleTasks) {
  // Dispatch multiple tasks on a single-threaded pool. Destruction drains all
  // pending tasks before the worker exits, so every task has run once the
  // runner has been destroyed.
  size_t NumTasksToRun = 10;
  std::atomic<size_t> TasksRun = 0;

  {
    ThreadPoolRunner R(1);
    for (size_t I = 0; I != NumTasksToRun; ++I)
      R([&]() { ++TasksRun; });
  }

  EXPECT_EQ(TasksRun, NumTasksToRun);
}

struct ConcurrencyState {
  std::future<int> FInit;
  std::promise<int> P1;
  std::promise<int> P2;
  std::future<int> F1 = P1.get_future();
  std::future<int> F2 = P2.get_future();
  std::promise<int> PResult;
};

TEST(ThreadPoolRunnerTest, ConcurrentTasks) {
  // Check that tasks run concurrently when multiple workers are available.
  // Two tasks communicate values back and forth via futures; neither can
  // complete without the other having started. FResult.get() also serves as
  // the "all tasks have run" wait point before destruction.
  std::promise<int> PInit;
  ConcurrencyState State;
  State.FInit = PInit.get_future();
  std::future<int> FResult = State.PResult.get_future();

  int ExpectedValue = 42;

  {
    ThreadPoolRunner R(2);
    R([&]() {
      State.P1.set_value(State.FInit.get());
      State.PResult.set_value(State.F2.get());
    });
    R([&]() { State.P2.set_value(State.F1.get()); });

    PInit.set_value(ExpectedValue);

    EXPECT_EQ(FResult.get(), ExpectedValue);
  }
}

} // namespace
