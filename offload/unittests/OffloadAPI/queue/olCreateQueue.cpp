//===------- Offload API tests - olCreateQueue ----------------------------===//
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
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using olCreateQueueTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olCreateQueueTest);

struct olCreateQueueMultiDeviceTest : OffloadDeviceTest {};
OFFLOAD_TESTS_INSTANTIATE_MULTI_DEVICE_FIXTURE(olCreateQueueMultiDeviceTest);

// Enqueues a host function that bumps Counter.
static ol_result_t launchIncrement(ol_queue_handle_t Queue,
                                   std::atomic<int> &Counter) {
  return olLaunchHostFunction(
      Queue, [](void *Data) { ++*static_cast<std::atomic<int> *>(Data); },
      &Counter);
}

TEST_P(olCreateQueueTest, Success) {
  ol_queue_handle_t Queue = nullptr;
  ASSERT_SUCCESS(olCreateQueue(Context, Device, &Queue));
  ASSERT_NE(Queue, nullptr);
  ASSERT_SUCCESS(olDestroyQueue(Queue));
}

TEST_P(olCreateQueueTest, InvalidNullHandleContext) {
  ol_queue_handle_t Queue = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olCreateQueue(nullptr, Device, &Queue));
}

TEST_P(olCreateQueueTest, InvalidNullHandleDevice) {
  ol_queue_handle_t Queue = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olCreateQueue(Context, nullptr, &Queue));
}

TEST_P(olCreateQueueTest, InvalidNullPointerQueue) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olCreateQueue(Context, Device, nullptr));
}

TEST_P(olCreateQueueTest, InvalidDeviceNotInContext) {
  ol_queue_handle_t Queue = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_DEVICE, olCreateQueue(Context, Host, &Queue));
}

// Several queues can be live at once on the same context and device; they are
// distinct and each one is usable.
TEST_P(olCreateQueueTest, SuccessMultipleQueues) {
  constexpr size_t NumQueues = 4;
  std::atomic<int> Counter{0};
  std::vector<ol_queue_handle_t> Queues(NumQueues, nullptr);

  for (size_t I = 0; I < NumQueues; ++I) {
    ASSERT_SUCCESS(olCreateQueue(Context, Device, &Queues[I]));
    ASSERT_NE(Queues[I], nullptr);
    for (size_t J = 0; J < I; ++J)
      ASSERT_NE(Queues[I], Queues[J]);
  }
  for (ol_queue_handle_t Queue : Queues) {
    ASSERT_SUCCESS(launchIncrement(Queue, Counter));
    ASSERT_SUCCESS(olSyncQueue(Queue));
    bool Empty = false;
    ASSERT_SUCCESS(
        olGetQueueInfo(Queue, OL_QUEUE_INFO_EMPTY, sizeof(Empty), &Empty));
    ASSERT_TRUE(Empty);
  }
  ASSERT_EQ(Counter.load(), static_cast<int>(NumQueues));
  for (ol_queue_handle_t Queue : Queues)
    ASSERT_SUCCESS(olDestroyQueue(Queue));
}

// Queues can be created and destroyed repeatedly.
TEST_P(olCreateQueueTest, SuccessCreateDestroyLoop) {
  for (int I = 0; I < 50; ++I) {
    ol_queue_handle_t Queue = nullptr;
    ASSERT_SUCCESS(olCreateQueue(Context, Device, &Queue));
    ASSERT_NE(Queue, nullptr);
    ASSERT_SUCCESS(olDestroyQueue(Queue));
  }
}

// A queue created after another one was destroyed with pending work is fully
// functional, whether or not the backend reuses the old resources.
TEST_P(olCreateQueueTest, SuccessAfterDestroyWithPendingWork) {
  SKIP_IF_FORCE_SYNC_OPS();

  ol_queue_handle_t OldQueue = nullptr;
  ASSERT_SUCCESS(olCreateQueue(Context, Device, &OldQueue));
  ManuallyTriggeredTask Task;
  ASSERT_SUCCESS(Task.enqueue(OldQueue));
  ASSERT_SUCCESS(olDestroyQueue(OldQueue));
  ASSERT_SUCCESS(Task.trigger());
  ASSERT_SUCCESS(olDestroyEvent(Task.CompleteEvent));

  std::atomic<int> Counter{0};
  ol_queue_handle_t NewQueue = nullptr;
  ASSERT_SUCCESS(olCreateQueue(Context, Device, &NewQueue));
  ASSERT_NE(NewQueue, nullptr);
  ASSERT_SUCCESS(launchIncrement(NewQueue, Counter));
  ASSERT_SUCCESS(olSyncQueue(NewQueue));
  ASSERT_EQ(Counter.load(), 1);
  bool Empty = false;
  ASSERT_SUCCESS(
      olGetQueueInfo(NewQueue, OL_QUEUE_INFO_EMPTY, sizeof(Empty), &Empty));
  ASSERT_TRUE(Empty);
  ASSERT_SUCCESS(olDestroyQueue(NewQueue));
}

// Destroying a queue leaves its context and device usable.
TEST_P(olCreateQueueTest, SuccessContextUsableAfterDestroy) {
  ol_queue_handle_t Queue = nullptr;
  ASSERT_SUCCESS(olCreateQueue(Context, Device, &Queue));
  ASSERT_SUCCESS(olDestroyQueue(Queue));

  size_t NumDevices = 0;
  ASSERT_SUCCESS(olGetContextInfo(Context, OL_CONTEXT_INFO_NUM_DEVICES,
                                  sizeof(NumDevices), &NumDevices));
  ASSERT_GE(NumDevices, 1u);
  ol_device_type_t Type;
  ASSERT_SUCCESS(
      olGetDeviceInfo(Device, OL_DEVICE_INFO_TYPE, sizeof(Type), &Type));

  ol_queue_handle_t Other = nullptr;
  ASSERT_SUCCESS(olCreateQueue(Context, Device, &Other));
  ASSERT_NE(Other, nullptr);
  ASSERT_SUCCESS(olDestroyQueue(Other));
}

// A context with several devices accepts one queue per device, each bound to
// its device.
TEST_P(olCreateQueueMultiDeviceTest, SuccessAllDevices) {
  std::vector<ol_queue_handle_t> Queues(Devices.size(), nullptr);

  for (size_t I = 0; I < Devices.size(); ++I) {
    SCOPED_TRACE("device index " + std::to_string(I));
    ASSERT_SUCCESS(olCreateQueue(Context, Devices[I], &Queues[I]));
    ASSERT_NE(Queues[I], nullptr);
    ol_device_handle_t Retrieved = nullptr;
    ASSERT_SUCCESS(olGetQueueInfo(Queues[I], OL_QUEUE_INFO_DEVICE,
                                  sizeof(Retrieved), &Retrieved));
    ASSERT_EQ(Retrieved, Devices[I]);
  }
  for (ol_queue_handle_t Queue : Queues)
    ASSERT_SUCCESS(olDestroyQueue(Queue));
}

// olCreateQueue and olDestroyQueue are thread safe when threads operate on
// their own queues.
TEST_P(olCreateQueueTest, ConcurrentCreateDestroy) {
  constexpr size_t NumThreads = 4;
  struct Result {
    ol_result_t Res = nullptr;
    bool NullHandle = false;
  };
  std::vector<Result> Results(NumThreads);
  std::vector<std::thread> Threads;
  for (size_t I = 0; I < NumThreads; ++I)
    Threads.emplace_back([&, I] {
      Result &R = Results[I];
      for (int Iter = 0; Iter < 25; ++Iter) {
        ol_queue_handle_t Queue = nullptr;
        if ((R.Res = olCreateQueue(Context, Device, &Queue)))
          return;
        if (!Queue) {
          R.NullHandle = true;
          return;
        }
        if ((R.Res = olDestroyQueue(Queue)))
          return;
      }
    });
  for (std::thread &T : Threads)
    T.join();

  for (const Result &R : Results) {
    ASSERT_SUCCESS(R.Res);
    ASSERT_FALSE(R.NullHandle);
  }
}

// Queues can be created, given work and destroyed concurrently while that
// work is still pending; the work completes afterwards.
TEST_P(olCreateQueueTest, ConcurrentCreateDestroyWithPendingWork) {
  SKIP_IF_FORCE_SYNC_OPS();

  constexpr size_t NumThreads = 4;
  std::vector<ManuallyTriggeredTask> Tasks(NumThreads);
  std::vector<ol_result_t> Results(NumThreads, nullptr);
  std::vector<std::thread> Threads;
  for (size_t I = 0; I < NumThreads; ++I)
    Threads.emplace_back([&, I] {
      ol_result_t &Res = Results[I];
      ol_queue_handle_t Queue = nullptr;
      if ((Res = olCreateQueue(Context, Device, &Queue)))
        return;
      if ((Res = Tasks[I].enqueue(Queue)))
        return;
      Res = olDestroyQueue(Queue);
    });
  for (std::thread &T : Threads)
    T.join();

  for (ol_result_t Res : Results)
    ASSERT_SUCCESS(Res);
  // Release every task before waiting; a backend may run them one at a time.
  for (ManuallyTriggeredTask &Task : Tasks) {
    std::lock_guard<std::mutex> Lock(Task.M);
    Task.Flag = true;
    Task.CV.notify_one();
  }
  for (ManuallyTriggeredTask &Task : Tasks) {
    ASSERT_SUCCESS(olSyncEvent(Task.CompleteEvent));
    ASSERT_SUCCESS(olDestroyEvent(Task.CompleteEvent));
  }
}
