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
#include <vector>

using namespace orc_rt;

namespace {

using TaskQueue = std::deque<move_only_function<void()>>;

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
  TaskQueue Q;
};

TEST_F(QueueingRunnerTest, EnqueueDoesNotRunImmediately) {
  QueueingRunner R(Q);
  R(dummySession(), /*CallId=*/0, dummyReturn(), recordingFn,
    WrapperFunctionBuffer());
  EXPECT_EQ(Log.size(), 0u) << "Enqueue should not run the call";
  EXPECT_EQ(Q.size(), 1u) << "Call should be sitting in the queue";
}

TEST_F(QueueingRunnerTest, RunFIFOUntilEmpty) {
  QueueingRunner R(Q);
  for (uint64_t I = 0; I < 3; ++I)
    R(dummySession(), I, dummyReturn(), recordingFn, WrapperFunctionBuffer());

  QueueingRunner<TaskQueue>::runFIFOUntilEmpty(Q);

  ASSERT_EQ(Log.size(), 3u);
  EXPECT_EQ(Log[0].CallId, 0u);
  EXPECT_EQ(Log[1].CallId, 1u);
  EXPECT_EQ(Log[2].CallId, 2u);
  EXPECT_TRUE(Q.empty());
}

TEST_F(QueueingRunnerTest, RunLIFOUntilEmpty) {
  QueueingRunner R(Q);
  for (uint64_t I = 0; I < 3; ++I)
    R(dummySession(), I, dummyReturn(), recordingFn, WrapperFunctionBuffer());

  QueueingRunner<TaskQueue>::runLIFOUntilEmpty(Q);

  ASSERT_EQ(Log.size(), 3u);
  EXPECT_EQ(Log[0].CallId, 2u);
  EXPECT_EQ(Log[1].CallId, 1u);
  EXPECT_EQ(Log[2].CallId, 0u);
  EXPECT_TRUE(Q.empty());
}

TEST_F(QueueingRunnerTest, DrainOnEmptyQueueIsNoOp) {
  // Both drain helpers should return immediately on an empty queue rather
  // than blocking.
  QueueingRunner<TaskQueue>::runFIFOUntilEmpty(Q);
  QueueingRunner<TaskQueue>::runLIFOUntilEmpty(Q);
  EXPECT_EQ(Log.size(), 0u);
}

TEST_F(QueueingRunnerTest, DrainPicksUpCallsEnqueuedDuringDrain) {
  // A call enqueued by a running call should also be drained in the same
  // runFIFOUntilEmpty call.
  QueueingRunner R(Q);

  // First call enqueues a second call from inside its body. We use a custom
  // wrapper-function (not recordingFn) to do that, since recordingFn doesn't
  // know about the queue.
  static QueueingRunner<TaskQueue> *PendingR = nullptr;
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

  QueueingRunner<TaskQueue>::runFIFOUntilEmpty(Q);

  ASSERT_EQ(Log.size(), 2u);
  EXPECT_EQ(Log[0].CallId, 0u);
  EXPECT_EQ(Log[1].CallId, 1u);
  PendingR = nullptr;
}

} // end anonymous namespace
