//===- QueueingTaskDispatcherTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt/QueueingTaskDispatcher.h"
#include "gtest/gtest.h"

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

using namespace orc_rt;

namespace {

TEST(QueueingTaskDispatcherTest, EmptyDispatcher) {
  // Test that a newly created dispatcher has no tasks
  QueueingTaskDispatcher Dispatcher;

  EXPECT_EQ(Dispatcher.pop_back(), nullptr);
  EXPECT_EQ(Dispatcher.pop_front(), nullptr);

  Dispatcher.shutdown();
}

TEST(QueueingTaskDispatcherTest, BasicTaskDispatch) {
  // Test basic task dispatching and retrieval
  QueueingTaskDispatcher Dispatcher;
  bool TaskRan = false;

  Dispatcher.dispatch(makeGenericTask([&]() { TaskRan = true; }));

  auto Task = Dispatcher.pop_back();
  EXPECT_NE(Task, nullptr);

  Task->run();
  EXPECT_TRUE(TaskRan);

  // Should be empty now
  EXPECT_EQ(Dispatcher.pop_back(), nullptr);
  EXPECT_EQ(Dispatcher.pop_front(), nullptr);

  Dispatcher.shutdown();
}

TEST(QueueingTaskDispatcherTest, MultipleTasks) {
  // Test dispatching multiple tasks
  QueueingTaskDispatcher Dispatcher;
  int TaskCount = 0;
  constexpr int NumTasks = 5;

  for (int I = 0; I < NumTasks; ++I)
    Dispatcher.dispatch(makeGenericTask([&]() { ++TaskCount; }));

  // Pop all tasks and run them
  for (int I = 0; I < NumTasks; ++I) {
    auto Task = Dispatcher.pop_back();
    EXPECT_NE(Task, nullptr);
    Task->run();
  }

  EXPECT_EQ(TaskCount, NumTasks);
  EXPECT_EQ(Dispatcher.pop_back(), nullptr);

  Dispatcher.shutdown();
}

TEST(QueueingTaskDispatcherTest, PopBackLIFOOrder) {
  // Test that pop_back retrieves tasks in LIFO (Last-In-First-Out) order
  QueueingTaskDispatcher Dispatcher;
  std::vector<int> ExecutionOrder;

  // Dispatch tasks with different values
  for (int I = 0; I < 3; ++I)
    Dispatcher.dispatch(makeGenericTask(
        [&ExecutionOrder, I]() { ExecutionOrder.push_back(I); }));

  // Pop from back should give us tasks in reverse order (LIFO)
  while (auto Task = Dispatcher.pop_back())
    Task->run();

  EXPECT_EQ(ExecutionOrder.size(), 3u);
  EXPECT_EQ(ExecutionOrder[0], 2); // Last dispatched task
  EXPECT_EQ(ExecutionOrder[1], 1);
  EXPECT_EQ(ExecutionOrder[2], 0); // First dispatched task

  Dispatcher.shutdown();
}

TEST(QueueingTaskDispatcherTest, PopFrontFIFOOrder) {
  // Test that pop_front retrieves tasks in FIFO (First-In-First-Out) order
  QueueingTaskDispatcher Dispatcher;
  std::vector<int> ExecutionOrder;

  // Dispatch tasks with different values
  for (int I = 0; I < 3; ++I)
    Dispatcher.dispatch(makeGenericTask(
        [&ExecutionOrder, I]() { ExecutionOrder.push_back(I); }));

  // Pop from front should give us tasks in original order (FIFO)
  while (auto Task = Dispatcher.pop_front())
    Task->run();

  EXPECT_EQ(ExecutionOrder.size(), 3u);
  EXPECT_EQ(ExecutionOrder[0], 0); // First dispatched task
  EXPECT_EQ(ExecutionOrder[1], 1);
  EXPECT_EQ(ExecutionOrder[2], 2); // Last dispatched task

  Dispatcher.shutdown();
}

TEST(QueueingTaskDispatcherTest, RunLIFOUntilEmpty) {
  // Test the runLIFOUntilEmpty method
  QueueingTaskDispatcher Dispatcher;
  std::vector<int> ExecutionOrder;

  // Dispatch tasks
  for (int I = 0; I < 3; ++I)
    Dispatcher.dispatch(makeGenericTask(
        [&ExecutionOrder, I]() { ExecutionOrder.push_back(I); }));

  // Run all tasks in LIFO order
  Dispatcher.runLIFOUntilEmpty();

  EXPECT_EQ(ExecutionOrder.size(), 3u);
  EXPECT_EQ(ExecutionOrder[0], 2); // Last dispatched task runs first
  EXPECT_EQ(ExecutionOrder[1], 1);
  EXPECT_EQ(ExecutionOrder[2], 0); // First dispatched task runs last

  // Should be empty now
  EXPECT_EQ(Dispatcher.pop_back(), nullptr);
  EXPECT_EQ(Dispatcher.pop_front(), nullptr);

  Dispatcher.shutdown();
}

TEST(QueueingTaskDispatcherTest, RunFIFOUntilEmpty) {
  // Test the runFIFOUntilEmpty method
  QueueingTaskDispatcher Dispatcher;
  std::vector<int> ExecutionOrder;

  // Dispatch tasks
  for (int I = 0; I < 3; ++I)
    Dispatcher.dispatch(makeGenericTask(
        [&ExecutionOrder, I]() { ExecutionOrder.push_back(I); }));

  // Run all tasks in FIFO order
  Dispatcher.runFIFOUntilEmpty();

  EXPECT_EQ(ExecutionOrder.size(), 3u);
  EXPECT_EQ(ExecutionOrder[0], 0); // First dispatched task runs first
  EXPECT_EQ(ExecutionOrder[1], 1);
  EXPECT_EQ(ExecutionOrder[2], 2); // Last dispatched task runs last

  // Should be empty now
  EXPECT_EQ(Dispatcher.pop_back(), nullptr);
  EXPECT_EQ(Dispatcher.pop_front(), nullptr);

  Dispatcher.shutdown();
}

TEST(QueueingTaskDispatcherTest, MixedPopOperations) {
  // Test mixing pop_front and pop_back operations
  QueueingTaskDispatcher Dispatcher;
  std::vector<int> ExecutionOrder;

  // Dispatch tasks 0, 1, 2
  for (int I = 0; I < 3; ++I)
    Dispatcher.dispatch(makeGenericTask(
        [&ExecutionOrder, I]() { ExecutionOrder.push_back(I); }));

  // Pop from back (should get task 2)
  auto Task1 = Dispatcher.pop_back();
  EXPECT_NE(Task1, nullptr);
  Task1->run();

  // Pop from front (should get task 0)
  auto Task2 = Dispatcher.pop_front();
  EXPECT_NE(Task2, nullptr);
  Task2->run();

  // Pop from back again (should get task 1)
  auto Task3 = Dispatcher.pop_back();
  EXPECT_NE(Task3, nullptr);
  Task3->run();

  // Should be empty now
  EXPECT_EQ(Dispatcher.pop_back(), nullptr);
  EXPECT_EQ(Dispatcher.pop_front(), nullptr);

  EXPECT_EQ(ExecutionOrder.size(), 3u);
  EXPECT_EQ(ExecutionOrder[0], 2); // Last task (pop_back)
  EXPECT_EQ(ExecutionOrder[1], 0); // First task (pop_front)
  EXPECT_EQ(ExecutionOrder[2], 1); // Middle task (pop_back)

  Dispatcher.shutdown();
}

TEST(QueueingTaskDispatcherTest, ShutdownWithPendingTasks) {
  // Test shutdown behavior when tasks remain in queue
  QueueingTaskDispatcher Dispatcher;

  // Dispatch some tasks but don't run them
  for (int I = 0; I < 3; ++I)
    Dispatcher.dispatch(makeGenericTask([]() {
      // These tasks won't be executed in this test
    }));

  // Should be able to shutdown even with pending tasks
  Dispatcher.shutdown();

  // After shutdown, no tasks should be available
  EXPECT_EQ(Dispatcher.pop_back(), nullptr);
  EXPECT_EQ(Dispatcher.pop_front(), nullptr);
}

TEST(QueueingTaskDispatcherTest, DispatchAfterShutdown) {
  // Test behavior of dispatch after shutdown
  QueueingTaskDispatcher Dispatcher;
  bool TaskRan = false;

  Dispatcher.shutdown();

  // Dispatch should work even after shutdown (tasks are queued)
  Dispatcher.dispatch(makeGenericTask([&]() { TaskRan = true; }));

  // Task should not be retrievable
  EXPECT_EQ(Dispatcher.pop_back(), nullptr);
  EXPECT_EQ(Dispatcher.pop_front(), nullptr);

  EXPECT_FALSE(TaskRan);
}

TEST(QueueingTaskDispatcherTest, RunMethodsOnEmptyDispatcher) {
  // Test that run methods work correctly on empty dispatcher
  QueueingTaskDispatcher Dispatcher;

  // These should not crash or hang
  Dispatcher.runLIFOUntilEmpty();
  Dispatcher.runFIFOUntilEmpty();

  Dispatcher.shutdown();
}

TEST(QueueingTaskDispatcherTest, InterleaveDispatchAndPop) {
  // Test interleaving dispatch and pop operations
  QueueingTaskDispatcher Dispatcher;
  std::vector<int> ExecutionOrder;

  // Dispatch task 0
  Dispatcher.dispatch(
      makeGenericTask([&ExecutionOrder]() { ExecutionOrder.push_back(0); }));

  // Pop and run task 0
  auto Task1 = Dispatcher.pop_back();
  EXPECT_NE(Task1, nullptr);
  Task1->run();

  // Dispatch tasks 1 and 2
  for (int I = 1; I < 3; ++I)
    Dispatcher.dispatch(makeGenericTask(
        [&ExecutionOrder, I]() { ExecutionOrder.push_back(I); }));

  // Pop and run remaining tasks
  while (auto Task = Dispatcher.pop_front())
    Task->run();

  EXPECT_EQ(ExecutionOrder.size(), 3u);
  EXPECT_EQ(ExecutionOrder[0], 0); // First task executed immediately
  EXPECT_EQ(ExecutionOrder[1], 1); // Second task (FIFO order)
  EXPECT_EQ(ExecutionOrder[2], 2); // Third task (FIFO order)

  Dispatcher.shutdown();
}

TEST(QueueingTaskDispatcherTest, ThreadSafety) {
  // Test thread safety of the dispatcher
  QueueingTaskDispatcher Dispatcher;
  constexpr int NumThreads = 4;
  constexpr int TasksPerThread = 25;
  std::atomic<int> TasksCompleted = 0;

  std::vector<std::thread> DispatchThreads;
  std::vector<std::thread> PopThreads;

  // Create threads that dispatch tasks
  for (int ThreadId = 0; ThreadId < NumThreads; ++ThreadId) {
    DispatchThreads.emplace_back([&]() {
      for (int I = 0; I < TasksPerThread; ++I) {
        Dispatcher.dispatch(makeGenericTask([&]() { ++TasksCompleted; }));
      }
    });
  }

  // Create threads that pop and run tasks
  for (int ThreadId = 0; ThreadId < NumThreads; ++ThreadId) {
    PopThreads.emplace_back([&]() {
      for (int I = 0; I < TasksPerThread; ++I) {
        std::unique_ptr<Task> Task;

        // Keep trying to pop a task
        while (!Task) {
          Task = Dispatcher.pop_back();
          if (!Task) {
            std::this_thread::yield();
          }
        }

        Task->run();
      }
    });
  }

  // Wait for all threads to complete
  for (auto &T : DispatchThreads)
    T.join();
  for (auto &T : PopThreads)
    T.join();

  EXPECT_EQ(TasksCompleted.load(), NumThreads * TasksPerThread);

  Dispatcher.shutdown();
}

TEST(QueueingTaskDispatcherTest, LargeNumberOfTasks) {
  // Test with a large number of tasks to ensure no performance issues
  QueueingTaskDispatcher Dispatcher;
  constexpr int NumTasks = 1000;
  int TasksRun = 0;

  // Dispatch many tasks
  for (int I = 0; I < NumTasks; ++I)
    Dispatcher.dispatch(makeGenericTask([&TasksRun]() { ++TasksRun; }));

  // Run all tasks using FIFO
  Dispatcher.runFIFOUntilEmpty();

  EXPECT_EQ(TasksRun, NumTasks);

  Dispatcher.shutdown();
}

} // end anonymous namespace
