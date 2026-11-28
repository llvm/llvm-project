//===------- Offload API tests - olSyncEvent -====-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using olSyncEventTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olSyncEventTest);

TEST_P(olSyncEventTest, Success) {
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));
  ASSERT_NE(Event, nullptr);
  ASSERT_SUCCESS(olSyncEvent(Event));
  ASSERT_SUCCESS(olDestroyEvent(Event));
}

TEST_P(olSyncEventTest, InvalidNullEvent) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olSyncEvent(nullptr));
}

// olSyncEvent does not consume the event and can be repeated.
TEST_P(olSyncEventTest, SuccessMultipleSync) {
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));
  ASSERT_NE(Event, nullptr);

  for (size_t I = 0; I < 10; I++)
    ASSERT_SUCCESS(olSyncEvent(Event));

  ASSERT_SUCCESS(olDestroyEvent(Event));
}

// olSyncEvent returns only after the work submitted before the event ran.
TEST_P(olSyncEventTest, SuccessBlocksUntilWorkCompletes) {
  SKIP_IF_FORCE_SYNC_OPS();

  std::atomic<bool> Ran{false};
  ManuallyTriggeredTask Task;
  ASSERT_SUCCESS(Task.enqueue(Queue));
  ASSERT_SUCCESS(enqueueSetFlag(Queue, Ran));
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));
  EXPECT_FALSE(Ran.load());

  // The task can only finish once another thread releases it.
  std::thread Releaser([&] { Task.release(); });
  ol_result_t Status = olSyncEvent(Event);
  bool RanAtReturn = Ran.load();
  Releaser.join();
  ASSERT_SUCCESS(Status);
  EXPECT_TRUE(RanAtReturn);

  ASSERT_SUCCESS(olDestroyEvent(Event));
  ASSERT_SUCCESS(olDestroyEvent(Task.CompleteEvent));
}

// A complete event can be synced again.
TEST_P(olSyncEventTest, SuccessCompleteEvent) {
  SKIP_IF_FORCE_SYNC_OPS();

  ManuallyTriggeredTask Task;
  ASSERT_SUCCESS(Task.enqueue(Queue));
  Task.release();
  ASSERT_SUCCESS(olSyncEvent(Task.CompleteEvent));
  ASSERT_SUCCESS(olSyncEvent(Task.CompleteEvent));
  ASSERT_SUCCESS(olDestroyEvent(Task.CompleteEvent));
}

// An event stays valid for olSyncEvent after its idle queue was destroyed.
TEST_P(olSyncEventTest, SuccessAfterDestroyIdleQueue) {
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_SUCCESS(olDestroyQueue(Queue));
  Queue = nullptr;
  ASSERT_SUCCESS(olSyncEvent(Event));
  ASSERT_SUCCESS(olDestroyEvent(Event));
}

// An event stays valid for olSyncEvent after its queue was destroyed with
// pending work, and still waits for that work.
TEST_P(olSyncEventTest, SuccessAfterDestroyBusyQueue) {
  SKIP_IF_FORCE_SYNC_OPS();

  std::atomic<bool> Ran{false};
  ManuallyTriggeredTask Task;
  ASSERT_SUCCESS(Task.enqueue(Queue));
  ASSERT_SUCCESS(enqueueSetFlag(Queue, Ran));
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));
  ASSERT_SUCCESS(olDestroyQueue(Queue));
  Queue = nullptr;
  Task.release();
  ASSERT_SUCCESS(olSyncEvent(Event));
  EXPECT_TRUE(Ran.load());
  ASSERT_SUCCESS(olDestroyEvent(Event));
  ASSERT_SUCCESS(olDestroyEvent(Task.CompleteEvent));
}

// A synced event can still be passed to olWaitEvents.
TEST_P(olSyncEventTest, SuccessWaitEventsAfterSync) {
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));
  ASSERT_SUCCESS(olSyncEvent(Event));

  ol_queue_handle_t Other = nullptr;
  ASSERT_SUCCESS(olCreateQueue(Context, Device, &Other));
  ASSERT_SUCCESS(olWaitEvents(Other, &Event, 1));
  ASSERT_SUCCESS(olSyncQueue(Other));
  ASSERT_SUCCESS(olDestroyQueue(Other));
  ASSERT_SUCCESS(olDestroyEvent(Event));
}

// An event can be passed to olWaitEvents on another queue after its own
// queue was destroyed.
TEST_P(olSyncEventTest, SuccessWaitEventsAfterDestroyQueue) {
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_SUCCESS(olDestroyQueue(Queue));
  Queue = nullptr;

  ol_queue_handle_t Other = nullptr;
  ASSERT_SUCCESS(olCreateQueue(Context, Device, &Other));
  ASSERT_SUCCESS(olWaitEvents(Other, &Event, 1));
  ASSERT_SUCCESS(olSyncQueue(Other));
  ASSERT_SUCCESS(olDestroyQueue(Other));
  ASSERT_SUCCESS(olDestroyEvent(Event));
}

// olSyncEvent is thread safe, including concurrent calls on the same event.
TEST_P(olSyncEventTest, ConcurrentSyncSameEvent) {
  constexpr size_t NumThreads = 4;
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Event));

  std::vector<ol_result_t> Results(NumThreads, nullptr);
  std::vector<std::thread> Threads;
  for (size_t I = 0; I < NumThreads; ++I)
    Threads.emplace_back([&, I] {
      for (int Iter = 0; Iter < 100; ++Iter)
        if ((Results[I] = olSyncEvent(Event)))
          return;
    });
  for (std::thread &T : Threads)
    T.join();

  for (ol_result_t Status : Results)
    ASSERT_SUCCESS(Status);
  ASSERT_SUCCESS(olDestroyEvent(Event));
}
