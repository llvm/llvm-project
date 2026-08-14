//===- QueueingRunnerTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt/QueueingRunner.h"
#include "orc-rt/move_only_function.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <thread>
#include <vector>

using namespace orc_rt;

namespace {

// Log of task ids, in the order the tasks ran. Tasks record their own id here
// when run, letting tests observe both whether and in what order tasks ran.
static std::vector<uint64_t> *RecordingLog = nullptr;

class QueueingRunnerTest : public ::testing::Test {
protected:
  void SetUp() override { RecordingLog = &Log; }
  void TearDown() override { RecordingLog = nullptr; }

  // Build a task that records the given id in the log when run.
  static move_only_function<void()> recordingTask(uint64_t Id) {
    return [Id]() { RecordingLog->push_back(Id); };
  }

  std::vector<uint64_t> Log;
  QueueingRunner<>::WorkQueue Q;
};

TEST_F(QueueingRunnerTest, EnqueueDoesNotRunImmediately) {
  QueueingRunner<> R(Q);
  R(recordingTask(0));
  EXPECT_EQ(Log.size(), 0u) << "Enqueue should not run the task";
  // Pop initial task.
  EXPECT_TRUE(Q.pop_back())
      << "At least one task should be sitting in the queue";
  EXPECT_FALSE(Q.pop_back())
      << "Exactly one task should have been sitting in the queue";
}

TEST_F(QueueingRunnerTest, RunFIFOUntilEmpty) {
  QueueingRunner R(Q);
  for (uint64_t I = 0; I < 3; ++I)
    R(recordingTask(I));

  QueueingRunner<>::runFIFOUntilEmpty(Q);

  ASSERT_EQ(Log.size(), 3u);
  EXPECT_EQ(Log[0], 0u);
  EXPECT_EQ(Log[1], 1u);
  EXPECT_EQ(Log[2], 2u);
  EXPECT_FALSE(Q.pop_back()); // Expect queue to be empty.
}

TEST_F(QueueingRunnerTest, RunLIFOUntilEmpty) {
  QueueingRunner R(Q);
  for (uint64_t I = 0; I < 3; ++I)
    R(recordingTask(I));

  QueueingRunner<>::runLIFOUntilEmpty(Q);

  ASSERT_EQ(Log.size(), 3u);
  EXPECT_EQ(Log[0], 2u);
  EXPECT_EQ(Log[1], 1u);
  EXPECT_EQ(Log[2], 0u);
  EXPECT_FALSE(Q.pop_back()); // Expect queue to be empty.
}

TEST_F(QueueingRunnerTest, DrainOnEmptyQueueIsNoOp) {
  // Both drain helpers should return immediately on an empty queue rather
  // than blocking.
  QueueingRunner<>::runFIFOUntilEmpty(Q);
  QueueingRunner<>::runLIFOUntilEmpty(Q);
  EXPECT_EQ(Log.size(), 0u);
}

TEST_F(QueueingRunnerTest, DrainPicksUpTasksEnqueuedDuringDrain) {
  // A task enqueued by a running task should also be drained in the same
  // runFIFOUntilEmpty call.
  QueueingRunner R(Q);

  // The first task enqueues a second task from inside its body.
  R([&]() {
    RecordingLog->push_back(0);
    R(recordingTask(1));
  });

  QueueingRunner<>::runFIFOUntilEmpty(Q);

  ASSERT_EQ(Log.size(), 2u);
  EXPECT_EQ(Log[0], 0u);
  EXPECT_EQ(Log[1], 1u);
}

TEST_F(QueueingRunnerTest, ConcurrentProducerAndDrainer) {
  // Verify that QueueingRunner's default WorkQueue (SynchronizedDeque)
  // tolerates concurrent push from one thread and drain from another.
  //
  // A producer thread enqueues NumTasks tasks while the main thread spins
  // draining the queue. Once the producer has finished enqueueing, the main
  // thread joins it and then performs a final drain to pick up any tail of
  // tasks enqueued after its last loop iteration.
  constexpr uint64_t NumTasks = 1024;

  QueueingRunner<> R(Q);

  std::thread Producer([&]() {
    for (uint64_t I = 0; I < NumTasks; ++I)
      R(recordingTask(I));
  });

  // Drain concurrently with the producer. The drainer doesn't know when the
  // producer is done, so we just spin until the producer thread has joined
  // (after which a final drain will be definitive).
  while (Log.size() < NumTasks) {
    QueueingRunner<>::runFIFOUntilEmpty(Q);
    std::this_thread::yield();
  }

  Producer.join();
  QueueingRunner<>::runFIFOUntilEmpty(Q); // pick up any tail.

  ASSERT_EQ(Log.size(), NumTasks);
  // Producer enqueues in order 0..NumTasks; FIFO drain must observe the same
  // order. (Concurrent draining doesn't reorder per-producer enqueues for a
  // single producer.)
  for (uint64_t I = 0; I < NumTasks; ++I)
    EXPECT_EQ(Log[I], I);

  EXPECT_FALSE(Q.pop_back()) << "Queue should be empty after final drain";
}

} // namespace
