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
#include <deque>
#include <thread>
#include <vector>

using namespace orc_rt;

namespace {

// A dummy SessionRef value used purely to thread an opaque pointer through
// the runner's enqueue path.
inline orc_rt_SessionRef dummySession() noexcept {
  return reinterpret_cast<orc_rt_SessionRef>(uintptr_t{0xABCD});
}

inline orc_rt_WrapperFunctionReturn dummyReturn() noexcept {
  return [](orc_rt_SessionRef, uint64_t, orc_rt_WrapperFunctionBuffer) {};
}

// Test wrapper-function that records each invocation in a globally-accessible
// log via its CallId.
struct CallRecord {
  orc_rt_SessionRef Session;
  uint64_t CallId;
};

static std::vector<CallRecord> *RecordingLog = nullptr;

static void recordingFn(orc_rt_SessionRef S, uint64_t CallId,
                        orc_rt_WrapperFunctionReturn,
                        orc_rt_WrapperFunctionBuffer ArgBytes) {
  WrapperFunctionBuffer Owned(ArgBytes);
  RecordingLog->push_back({S, CallId});
}

class QueueingRunnerTest : public ::testing::Test {
protected:
  void SetUp() override { RecordingLog = &Log; }
  void TearDown() override { RecordingLog = nullptr; }

  std::vector<CallRecord> Log;
  QueueingRunner<>::WorkQueue Q;
};

TEST_F(QueueingRunnerTest, EnqueueDoesNotRunImmediately) {
  QueueingRunner<> R(Q);
  R(dummySession(), /*CallId=*/0, dummyReturn(), recordingFn,
    WrapperFunctionBuffer());
  EXPECT_EQ(Log.size(), 0u) << "Enqueue should not run the call";
  // Pop initial call.
  EXPECT_TRUE(Q.pop_back())
      << "At least one call should be sitting in the queue";
  EXPECT_FALSE(Q.pop_back())
      << "Exactly one call should have been sitting in the queue";
}

TEST_F(QueueingRunnerTest, RunFIFOUntilEmpty) {
  QueueingRunner R(Q);
  for (uint64_t I = 0; I < 3; ++I)
    R(dummySession(), I, dummyReturn(), recordingFn, WrapperFunctionBuffer());

  QueueingRunner<>::runFIFOUntilEmpty(Q);

  ASSERT_EQ(Log.size(), 3u);
  EXPECT_EQ(Log[0].CallId, 0u);
  EXPECT_EQ(Log[1].CallId, 1u);
  EXPECT_EQ(Log[2].CallId, 2u);
  EXPECT_FALSE(Q.pop_back()); // Expect queue to be empty.
}

TEST_F(QueueingRunnerTest, RunLIFOUntilEmpty) {
  QueueingRunner R(Q);
  for (uint64_t I = 0; I < 3; ++I)
    R(dummySession(), I, dummyReturn(), recordingFn, WrapperFunctionBuffer());

  QueueingRunner<>::runLIFOUntilEmpty(Q);

  ASSERT_EQ(Log.size(), 3u);
  EXPECT_EQ(Log[0].CallId, 2u);
  EXPECT_EQ(Log[1].CallId, 1u);
  EXPECT_EQ(Log[2].CallId, 0u);
  EXPECT_FALSE(Q.pop_back()); // Expect queue to be empty.
}

TEST_F(QueueingRunnerTest, DrainOnEmptyQueueIsNoOp) {
  // Both drain helpers should return immediately on an empty queue rather
  // than blocking.
  QueueingRunner<>::runFIFOUntilEmpty(Q);
  QueueingRunner<>::runLIFOUntilEmpty(Q);
  EXPECT_EQ(Log.size(), 0u);
}

TEST_F(QueueingRunnerTest, DrainPicksUpCallsEnqueuedDuringDrain) {
  // A call enqueued by a running call should also be drained in the same
  // runFIFOUntilEmpty call.
  QueueingRunner R(Q);

  // First call enqueues a second call from inside its body. We use a custom
  // wrapper-function (not recordingFn) to do that, since recordingFn doesn't
  // know about the queue.
  static QueueingRunner<> *PendingR = nullptr;
  PendingR = &R;
  static auto reentrantFn = [](orc_rt_SessionRef S, uint64_t CallId,
                               orc_rt_WrapperFunctionReturn,
                               orc_rt_WrapperFunctionBuffer ArgBytes) {
    WrapperFunctionBuffer Owned(ArgBytes);
    RecordingLog->push_back({S, CallId});
    if (CallId == 0)
      (*PendingR)(S, /*CallId=*/1, dummyReturn(), recordingFn,
                  WrapperFunctionBuffer());
  };

  R(dummySession(), /*CallId=*/0, dummyReturn(), reentrantFn,
    WrapperFunctionBuffer());

  QueueingRunner<>::runFIFOUntilEmpty(Q);

  ASSERT_EQ(Log.size(), 2u);
  EXPECT_EQ(Log[0].CallId, 0u);
  EXPECT_EQ(Log[1].CallId, 1u);
  PendingR = nullptr;
}

TEST_F(QueueingRunnerTest, ConcurrentProducerAndDrainer) {
  // Verify that QueueingRunner's default WorkQueue (SynchronizedDeque)
  // tolerates concurrent push from one thread and drain from another.
  //
  // A producer thread enqueues NumCalls wrapper-function invocations while
  // the main thread spins draining the queue. Once the producer has finished
  // enqueueing, the main thread joins it and then performs a final drain to
  // pick up any tail of calls enqueued after its last loop iteration.
  constexpr uint64_t NumCalls = 1024;

  QueueingRunner<> R(Q);

  std::thread Producer([&]() {
    for (uint64_t I = 0; I < NumCalls; ++I)
      R(dummySession(), I, dummyReturn(), recordingFn, WrapperFunctionBuffer());
  });

  // Drain concurrently with the producer. The drainer doesn't know when the
  // producer is done, so we just spin until the producer thread has joined
  // (after which a final drain will be definitive).
  while (Log.size() < NumCalls) {
    QueueingRunner<>::runFIFOUntilEmpty(Q);
    std::this_thread::yield();
  }

  Producer.join();
  QueueingRunner<>::runFIFOUntilEmpty(Q); // pick up any tail.

  ASSERT_EQ(Log.size(), NumCalls);
  // Producer enqueues in order 0..NumCalls; FIFO drain must observe the same
  // order. (Concurrent draining doesn't reorder per-producer enqueues for a
  // single producer.)
  for (uint64_t I = 0; I < NumCalls; ++I)
    EXPECT_EQ(Log[I].CallId, I);

  EXPECT_FALSE(Q.pop_back()) << "Queue should be empty after final drain";
}

} // end anonymous namespace
