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

TEST(QueueingTaskDispatcherTest, BasicTaskDispatch) {
  // Test basic task dispatching and retrieval.
  QueueingTaskDispatcher::TaskQueue Q;
  QueueingTaskDispatcher Dispatcher(Q);
  bool TaskRan = false;

  Dispatcher.dispatch(makeGenericTask([&]() { TaskRan = true; }));
  Dispatcher.shutdown();

  auto Task = Q.takeFirstIn();
  EXPECT_NE(Task, nullptr);
  Task->run();
  EXPECT_TRUE(TaskRan);

  // Queue is shut down and drained — should return nullptr.
  EXPECT_EQ(Q.takeFirstIn(), nullptr);
}

TEST(QueueingTaskDispatcherTest, MultipleTasks) {
  // Test dispatching and running multiple tasks.
  QueueingTaskDispatcher::TaskQueue Q;
  QueueingTaskDispatcher Dispatcher(Q);
  int TaskCount = 0;
  constexpr int NumTasks = 5;

  for (int I = 0; I < NumTasks; ++I)
    Dispatcher.dispatch(makeGenericTask([&]() { ++TaskCount; }));
  Dispatcher.shutdown();

  // Take and run all tasks.
  for (int I = 0; I < NumTasks; ++I) {
    auto Task = Q.takeFirstIn();
    EXPECT_NE(Task, nullptr);
    Task->run();
  }

  EXPECT_EQ(TaskCount, NumTasks);
  EXPECT_EQ(Q.takeFirstIn(), nullptr);
}

TEST(QueueingTaskDispatcherTest, TakeLastInLIFOOrder) {
  // Test that takeLastIn retrieves tasks in LIFO order.
  QueueingTaskDispatcher::TaskQueue Q;
  QueueingTaskDispatcher Dispatcher(Q);
  std::vector<int> ExecutionOrder;

  for (int I = 0; I < 3; ++I)
    Dispatcher.dispatch(makeGenericTask(
        [&ExecutionOrder, I]() { ExecutionOrder.push_back(I); }));
  Dispatcher.shutdown();

  while (auto Task = Q.takeLastIn())
    Task->run();

  ASSERT_EQ(ExecutionOrder.size(), 3u);
  EXPECT_EQ(ExecutionOrder[0], 2);
  EXPECT_EQ(ExecutionOrder[1], 1);
  EXPECT_EQ(ExecutionOrder[2], 0);
}

TEST(QueueingTaskDispatcherTest, TakeFirstInFIFOOrder) {
  // Test that takeFirstIn retrieves tasks in FIFO order.
  QueueingTaskDispatcher::TaskQueue Q;
  QueueingTaskDispatcher Dispatcher(Q);
  std::vector<int> ExecutionOrder;

  for (int I = 0; I < 3; ++I)
    Dispatcher.dispatch(makeGenericTask(
        [&ExecutionOrder, I]() { ExecutionOrder.push_back(I); }));
  Dispatcher.shutdown();

  while (auto Task = Q.takeFirstIn())
    Task->run();

  ASSERT_EQ(ExecutionOrder.size(), 3u);
  EXPECT_EQ(ExecutionOrder[0], 0);
  EXPECT_EQ(ExecutionOrder[1], 1);
  EXPECT_EQ(ExecutionOrder[2], 2);
}

TEST(QueueingTaskDispatcherTest, RunLIFOUntilEmpty) {
  // Test the runLIFOUntilEmpty convenience method.
  QueueingTaskDispatcher::TaskQueue Q;
  QueueingTaskDispatcher Dispatcher(Q);
  std::vector<int> ExecutionOrder;

  for (int I = 0; I < 3; ++I)
    Dispatcher.dispatch(makeGenericTask(
        [&ExecutionOrder, I]() { ExecutionOrder.push_back(I); }));
  Dispatcher.shutdown();

  Q.runLIFOUntilEmpty();

  ASSERT_EQ(ExecutionOrder.size(), 3u);
  EXPECT_EQ(ExecutionOrder[0], 2);
  EXPECT_EQ(ExecutionOrder[1], 1);
  EXPECT_EQ(ExecutionOrder[2], 0);
}

TEST(QueueingTaskDispatcherTest, RunFIFOUntilEmpty) {
  // Test the runFIFOUntilEmpty convenience method.
  QueueingTaskDispatcher::TaskQueue Q;
  QueueingTaskDispatcher Dispatcher(Q);
  std::vector<int> ExecutionOrder;

  for (int I = 0; I < 3; ++I)
    Dispatcher.dispatch(makeGenericTask(
        [&ExecutionOrder, I]() { ExecutionOrder.push_back(I); }));
  Dispatcher.shutdown();

  Q.runFIFOUntilEmpty();

  ASSERT_EQ(ExecutionOrder.size(), 3u);
  EXPECT_EQ(ExecutionOrder[0], 0);
  EXPECT_EQ(ExecutionOrder[1], 1);
  EXPECT_EQ(ExecutionOrder[2], 2);
}

TEST(QueueingTaskDispatcherTest, MixedTakeOperations) {
  // Test mixing takeFirstIn and takeLastIn.
  QueueingTaskDispatcher::TaskQueue Q;
  QueueingTaskDispatcher Dispatcher(Q);
  std::vector<int> ExecutionOrder;

  // Dispatch tasks 0, 1, 2.
  for (int I = 0; I < 3; ++I)
    Dispatcher.dispatch(makeGenericTask(
        [&ExecutionOrder, I]() { ExecutionOrder.push_back(I); }));
  Dispatcher.shutdown();

  // takeLastIn should get task 2.
  auto Task1 = Q.takeLastIn();
  ASSERT_NE(Task1, nullptr);
  Task1->run();

  // takeFirstIn should get task 0.
  auto Task2 = Q.takeFirstIn();
  ASSERT_NE(Task2, nullptr);
  Task2->run();

  // takeLastIn should get task 1 (only one left).
  auto Task3 = Q.takeLastIn();
  ASSERT_NE(Task3, nullptr);
  Task3->run();

  EXPECT_EQ(Q.takeFirstIn(), nullptr);

  ASSERT_EQ(ExecutionOrder.size(), 3u);
  EXPECT_EQ(ExecutionOrder[0], 2);
  EXPECT_EQ(ExecutionOrder[1], 0);
  EXPECT_EQ(ExecutionOrder[2], 1);
}

TEST(QueueingTaskDispatcherTest, ShutdownDrainsRemainingTasks) {
  // Verify that tasks dispatched before shutdown can still be taken.
  QueueingTaskDispatcher::TaskQueue Q;
  QueueingTaskDispatcher Dispatcher(Q);
  int TaskCount = 0;

  for (int I = 0; I < 3; ++I)
    Dispatcher.dispatch(makeGenericTask([&]() { ++TaskCount; }));

  Dispatcher.shutdown();

  // All pre-shutdown tasks should still be available.
  while (auto Task = Q.takeFirstIn())
    Task->run();

  EXPECT_EQ(TaskCount, 3);
}

TEST(QueueingTaskDispatcherTest, DispatchAfterShutdown) {
  // Tasks dispatched after shutdown should be discarded.
  QueueingTaskDispatcher::TaskQueue Q;
  QueueingTaskDispatcher Dispatcher(Q);
  bool TaskRan = false;

  Dispatcher.shutdown();

  Dispatcher.dispatch(makeGenericTask([&]() { TaskRan = true; }));

  EXPECT_EQ(Q.takeFirstIn(), nullptr);
  EXPECT_FALSE(TaskRan);
}

TEST(QueueingTaskDispatcherTest, TakeBlocksUntilTaskAvailable) {
  // Verify that takeFirstIn blocks on an empty queue until a task arrives.
  QueueingTaskDispatcher::TaskQueue Q;
  QueueingTaskDispatcher Dispatcher(Q);
  std::atomic<bool> TaskTaken = false;

  std::thread Consumer([&]() {
    auto Task = Q.takeFirstIn();
    TaskTaken = true;
    EXPECT_NE(Task, nullptr);
    Task->run();
  });

  // Give the consumer a moment to block.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(TaskTaken);

  // Dispatching a task should unblock the consumer.
  std::atomic<bool> TaskRan = false;
  Dispatcher.dispatch(makeGenericTask([&]() { TaskRan = true; }));

  Consumer.join();

  EXPECT_TRUE(TaskTaken);
  EXPECT_TRUE(TaskRan);

  Dispatcher.shutdown();
}

TEST(QueueingTaskDispatcherTest, TakeReturnsNullptrOnShutdown) {
  // Verify that a blocked take returns nullptr when the queue is shut down.
  QueueingTaskDispatcher::TaskQueue Q;
  QueueingTaskDispatcher Dispatcher(Q);
  std::atomic<bool> TakeReturned = false;

  std::thread Consumer([&]() {
    auto Task = Q.takeFirstIn();
    EXPECT_EQ(Task, nullptr);
    TakeReturned.store(true);
  });

  // Give the consumer a moment to block.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(TakeReturned);

  // Shutting down should unblock the consumer with nullptr.
  Dispatcher.shutdown();

  Consumer.join();
  EXPECT_TRUE(TakeReturned);
}

TEST(QueueingTaskDispatcherTest, ThreadSafety) {
  // Test thread safety with concurrent dispatch and take.
  QueueingTaskDispatcher::TaskQueue Q;
  QueueingTaskDispatcher Dispatcher(Q);
  constexpr int NumProducers = 4;
  constexpr int TasksPerProducer = 25;
  constexpr int TotalTasks = NumProducers * TasksPerProducer;
  std::atomic<int> TasksCompleted = 0;

  // Producer threads dispatch tasks.
  std::vector<std::thread> Producers;
  for (int I = 0; I < NumProducers; ++I) {
    Producers.emplace_back([&]() {
      for (int J = 0; J < TasksPerProducer; ++J)
        Dispatcher.dispatch(makeGenericTask([&]() { ++TasksCompleted; }));
    });
  }

  // Consumer thread takes and runs tasks until shutdown.
  std::thread Consumer([&]() {
    while (auto Task = Q.takeFirstIn())
      Task->run();
  });

  // Wait for all producers to finish, then shut down.
  for (auto &T : Producers)
    T.join();
  Dispatcher.shutdown();

  Consumer.join();
  EXPECT_EQ(TasksCompleted, TotalTasks);
}

} // end anonymous namespace
