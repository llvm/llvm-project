//===------- Offload API tests - olDestroyEvent ---------------------------===//
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

using olDestroyEventTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olDestroyEventTest);

// A completed event can be destroyed.
TEST_P(olDestroyEventTest, Success) {
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));
  ASSERT_NE(Event, nullptr);
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_SUCCESS(olDestroyEvent(Event));
}

TEST_P(olDestroyEventTest, InvalidNullEvent) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olDestroyEvent(nullptr));
}

// Destroying an incomplete event neither waits for it nor cancels the work:
// the call returns while the work is blocked, and the work still completes.
TEST_P(olDestroyEventTest, SuccessWithPendingWork) {
  SKIP_IF_FORCE_SYNC_OPS();

  std::atomic<bool> Ran{false};
  ManuallyTriggeredTask Task;
  ASSERT_SUCCESS(Task.enqueue(Queue));
  ASSERT_SUCCESS(enqueueSetFlag(Queue, Ran));
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));

  ASSERT_SUCCESS(olDestroyEvent(Event));

  Task.release();
  ASSERT_SUCCESS(olSyncQueue(Queue));
  EXPECT_TRUE(Ran.load());
  ASSERT_SUCCESS(olDestroyEvent(Task.CompleteEvent));
}

// An event remains valid for olDestroyEvent after its queue was destroyed.
// The queue is idle when destroyed.
TEST_P(olDestroyEventTest, SuccessAfterDestroyQueue) {
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_SUCCESS(olDestroyQueue(Queue));
  Queue = nullptr;
  ASSERT_SUCCESS(olDestroyEvent(Event));
}

// An event remains valid for olDestroyEvent after its queue was destroyed with
// pending work.
TEST_P(olDestroyEventTest, SuccessAfterDestroyBusyQueue) {
  SKIP_IF_FORCE_SYNC_OPS();

  ManuallyTriggeredTask Task;
  ASSERT_SUCCESS(Task.enqueue(Queue));
  ASSERT_SUCCESS(olDestroyQueue(Queue));
  Queue = nullptr;
  Task.release();
  ASSERT_SUCCESS(olSyncEvent(Task.CompleteEvent));
  ASSERT_SUCCESS(olDestroyEvent(Task.CompleteEvent));
}

// olDestroyEvent is thread safe when threads destroy distinct events.
TEST_P(olDestroyEventTest, ConcurrentDestroyDistinctEvents) {
  constexpr size_t NumThreads = 4;
  constexpr size_t PerThread = 25;
  std::vector<ol_event_handle_t> Events(NumThreads * PerThread, nullptr);
  for (ol_event_handle_t &Event : Events)
    ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));
  ASSERT_SUCCESS(olSyncQueue(Queue));

  std::vector<ol_result_t> Results(NumThreads, nullptr);
  std::vector<std::thread> Threads;
  for (size_t I = 0; I < NumThreads; ++I)
    Threads.emplace_back([&, I] {
      for (size_t J = 0; J < PerThread; ++J)
        if ((Results[I] = olDestroyEvent(Events[I * PerThread + J])))
          return;
    });
  for (std::thread &T : Threads)
    T.join();

  for (ol_result_t Status : Results)
    ASSERT_SUCCESS(Status);
}
