//===------- Offload API tests - olCreateEvent ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using olCreateEventTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olCreateEventTest);

// The event returned by olCreateEvent is released with olDestroyEvent.
TEST_P(olCreateEventTest, Success) {
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));
  ASSERT_NE(Event, nullptr);
  ASSERT_SUCCESS(olDestroyEvent(Event));
}

// An event created with OL_EVENT_FLAGS_ENABLE_PROFILING can be synced and
// destroyed.
TEST_P(olCreateEventTest, SuccessWithProfiling) {
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_ENABLE_PROFILING, &Event));
  ASSERT_NE(Event, nullptr);
  ASSERT_SUCCESS(olSyncEvent(Event));
  ASSERT_SUCCESS(olDestroyEvent(Event));
}

// A failed olCreateEvent does not write the output.
TEST_P(olCreateEventTest, InvalidNullQueue) {
  ol_event_handle_t Sentinel = reinterpret_cast<ol_event_handle_t>(0x1);
  ol_event_handle_t Event = Sentinel;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olCreateEvent(nullptr, OL_EVENT_FLAGS_NONE, &Event));
  ASSERT_EQ(Event, Sentinel);
}

TEST_P(olCreateEventTest, InvalidNullDest) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, nullptr));
}

// An event created on a queue with no pending work is complete without a sync.
TEST_P(olCreateEventTest, SuccessEmptyQueueIsComplete) {
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));
  bool Complete = false;
  ASSERT_SUCCESS(olGetEventInfo(Event, OL_EVENT_INFO_IS_COMPLETE,
                                sizeof(Complete), &Complete));
  ASSERT_TRUE(Complete);
  ASSERT_SUCCESS(olSyncEvent(Event));
  ASSERT_SUCCESS(olDestroyEvent(Event));
}

// The event covers work submitted before the call: it is complete only after
// that work ran.
TEST_P(olCreateEventTest, SuccessCoversPendingWork) {
  SKIP_IF_FORCE_SYNC_OPS();

  std::atomic<bool> Ran{false};
  ManuallyTriggeredTask Task;
  ASSERT_SUCCESS(Task.enqueue(Queue));
  ASSERT_SUCCESS(enqueueSetFlag(Queue, Ran));
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));

  Task.release();
  ASSERT_SUCCESS(olSyncEvent(Event));
  EXPECT_TRUE(Ran.load());
  bool Complete = false;
  ASSERT_SUCCESS(olGetEventInfo(Event, OL_EVENT_INFO_IS_COMPLETE,
                                sizeof(Complete), &Complete));
  EXPECT_TRUE(Complete);
  ASSERT_SUCCESS(olDestroyEvent(Event));
  ASSERT_SUCCESS(olDestroyEvent(Task.CompleteEvent));
}

// Several events can be live on one queue; they are distinct.
TEST_P(olCreateEventTest, SuccessMultipleEvents) {
  constexpr size_t NumEvents = 4;
  std::vector<ol_event_handle_t> Events(NumEvents, nullptr);
  for (size_t I = 0; I < NumEvents; ++I) {
    ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Events[I]));
    ASSERT_NE(Events[I], nullptr);
    for (size_t J = 0; J < I; ++J)
      ASSERT_NE(Events[I], Events[J]);
  }
  for (ol_event_handle_t Event : Events) {
    ASSERT_SUCCESS(olSyncEvent(Event));
    ASSERT_SUCCESS(olDestroyEvent(Event));
  }
}

// olCreateEvent is thread safe, including on the same queue.
TEST_P(olCreateEventTest, ConcurrentCreateSameQueue) {
  constexpr size_t NumThreads = 4;
  struct Result {
    ol_result_t Status = nullptr;
    bool NullHandle = false;
  };
  std::vector<Result> Results(NumThreads);
  std::vector<std::thread> Threads;
  for (size_t I = 0; I < NumThreads; ++I)
    Threads.emplace_back([&, I] {
      Result &R = Results[I];
      for (int Iter = 0; Iter < 25; ++Iter) {
        ol_event_handle_t Event = nullptr;
        if ((R.Status = olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event)))
          return;
        if (!Event) {
          R.NullHandle = true;
          return;
        }
        if ((R.Status = olDestroyEvent(Event)))
          return;
      }
    });
  for (std::thread &T : Threads)
    T.join();

  for (const Result &R : Results) {
    ASSERT_SUCCESS(R.Status);
    ASSERT_FALSE(R.NullHandle);
  }
}

// olCreateEvent is thread safe when threads use separate queues.
TEST_P(olCreateEventTest, ConcurrentCreateSeparateQueues) {
  constexpr size_t NumThreads = 4;
  std::vector<ol_result_t> Results(NumThreads, nullptr);
  std::vector<std::thread> Threads;
  for (size_t I = 0; I < NumThreads; ++I)
    Threads.emplace_back([&, I] {
      ol_result_t &Status = Results[I];
      ol_queue_handle_t OwnQueue = nullptr;
      if ((Status = olCreateQueue(Context, Device, &OwnQueue)))
        return;
      for (int Iter = 0; Iter < 25; ++Iter) {
        ol_event_handle_t Event = nullptr;
        if ((Status = olCreateEvent(OwnQueue, OL_EVENT_FLAGS_NONE, &Event)))
          return;
        if ((Status = olSyncEvent(Event)))
          return;
        if ((Status = olDestroyEvent(Event)))
          return;
      }
      Status = olDestroyQueue(OwnQueue);
    });
  for (std::thread &T : Threads)
    T.join();

  for (ol_result_t Status : Results)
    ASSERT_SUCCESS(Status);
}
