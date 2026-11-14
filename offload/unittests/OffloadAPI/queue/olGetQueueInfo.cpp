//===------- Offload API tests - olGetQueueInfo ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include <algorithm>
#include <map>
#include <vector>

#include "../common/Fixtures.hpp"

#include <thread>
#include <vector>

using olGetQueueInfoTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetQueueInfoTest);

TEST_P(olGetQueueInfoTest, SuccessDevice) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_SUCCESS(olGetQueueInfo(Queue, OL_QUEUE_INFO_DEVICE,
                                sizeof(ol_device_handle_t), &RetrievedDevice));
  ASSERT_EQ(Device, RetrievedDevice);
}

TEST_P(olGetQueueInfoTest, SuccessContext) {
  ol_context_handle_t RetrievedContext;
  ASSERT_SUCCESS(olGetQueueInfo(Queue, OL_QUEUE_INFO_CONTEXT,
                                sizeof(ol_context_handle_t),
                                &RetrievedContext));
  ASSERT_EQ(Context, RetrievedContext);
}

// A queue with no submitted work is empty.
TEST_P(olGetQueueInfoTest, SuccessEmpty) {
  bool Empty = false;
  ASSERT_SUCCESS(
      olGetQueueInfo(Queue, OL_QUEUE_INFO_EMPTY, sizeof(Empty), &Empty));
  ASSERT_TRUE(Empty);
}

// EMPTY is false while submitted work is outstanding and true once that work
// has completed and the queue was synchronized.
TEST_P(olGetQueueInfoTest, SuccessEmptyWithPendingWork) {
  SKIP_IF_FORCE_SYNC_OPS();

  ManuallyTriggeredTask Task;
  ASSERT_SUCCESS(Task.enqueue(Queue));

  bool Empty = true;
  ASSERT_SUCCESS(
      olGetQueueInfo(Queue, OL_QUEUE_INFO_EMPTY, sizeof(Empty), &Empty));
  EXPECT_FALSE(Empty);

  ASSERT_SUCCESS(Task.trigger());
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_SUCCESS(
      olGetQueueInfo(Queue, OL_QUEUE_INFO_EMPTY, sizeof(Empty), &Empty));
  EXPECT_TRUE(Empty);
  ASSERT_SUCCESS(olDestroyEvent(Task.CompleteEvent));
}

// CONTEXT returns the context that was passed to olCreateQueue, also when it
// is not the fixture's context.
TEST_P(olGetQueueInfoTest, SuccessSecondContext) {
  ol_context_handle_t OtherContext = nullptr;
  ASSERT_SUCCESS(olCreateContext(1, &Device, &OtherContext));
  ol_queue_handle_t OtherQueue = nullptr;
  ASSERT_SUCCESS(olCreateQueue(OtherContext, Device, &OtherQueue));

  ol_context_handle_t Retrieved = nullptr;
  ASSERT_SUCCESS(olGetQueueInfo(OtherQueue, OL_QUEUE_INFO_CONTEXT,
                                sizeof(Retrieved), &Retrieved));
  ASSERT_EQ(Retrieved, OtherContext);
  ASSERT_NE(Retrieved, Context);

  ASSERT_SUCCESS(olDestroyQueue(OtherQueue));
  ASSERT_SUCCESS(olDestroyContext(OtherContext));
}

TEST_P(olGetQueueInfoTest, SuccessNativeHandlesSpread) {
  // Queues created while all others are still live should be spread evenly
  // over whatever hardware queues the device multiplexes them onto: the
  // busiest and the idlest hardware queue differ by at most one user.
  constexpr size_t NumQueues = 8;
  std::vector<ol_queue_handle_t> Queues(NumQueues, nullptr);
  std::map<void *, size_t> Uses;
  for (auto &Q : Queues) {
    ASSERT_SUCCESS(olCreateQueue(Context, Device, &Q));
    void *Handle = nullptr;
    ol_result_t Result =
        olGetQueueInfo(Q, OL_QUEUE_INFO_NATIVE_HANDLE, sizeof(Handle), &Handle);
    if (Result && Result->Code == OL_ERRC_UNSUPPORTED) {
      for (auto &Other : Queues)
        if (Other)
          ASSERT_SUCCESS(olDestroyQueue(Other));
      GTEST_SKIP() << "Native queue handles are not supported";
    }
    ASSERT_SUCCESS(Result);
    ASSERT_NE(Handle, nullptr);
    ++Uses[Handle];
  }

  auto [Min, Max] = std::minmax_element(
      Uses.begin(), Uses.end(),
      [](const auto &A, const auto &B) { return A.second < B.second; });
  EXPECT_LE(Max->second - Min->second, 1u)
      << Uses.size() << " hardware queues for " << NumQueues << " queues";

  for (auto &Q : Queues)
    ASSERT_SUCCESS(olDestroyQueue(Q));
}

TEST_P(olGetQueueInfoTest, InvalidNullHandle) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetQueueInfo(nullptr, OL_QUEUE_INFO_DEVICE,
                              sizeof(RetrievedDevice), &RetrievedDevice));
}

TEST_P(olGetQueueInfoTest, InvalidQueueInfoEnumeration) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetQueueInfo(Queue, OL_QUEUE_INFO_FORCE_UINT32,
                              sizeof(RetrievedDevice), &RetrievedDevice));
}

TEST_P(olGetQueueInfoTest, InvalidSizeZero) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE, olGetQueueInfo(Queue, OL_QUEUE_INFO_DEVICE,
                                                    0, &RetrievedDevice));
}

TEST_P(olGetQueueInfoTest, InvalidSizeSmall) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetQueueInfo(Queue, OL_QUEUE_INFO_DEVICE,
                              sizeof(RetrievedDevice) - 1, &RetrievedDevice));
}

TEST_P(olGetQueueInfoTest, InvalidNullPointerPropValue) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetQueueInfo(Queue, OL_QUEUE_INFO_DEVICE,
                              sizeof(RetrievedDevice), nullptr));
}

TEST_P(olGetQueueInfoTest, SuccessNativeHandle) {
  void *Handle = nullptr;
  ASSERT_SUCCESS_OR_UNSUPPORTED(olGetQueueInfo(
      Queue, OL_QUEUE_INFO_NATIVE_HANDLE, sizeof(Handle), &Handle));
  ASSERT_NE(Handle, nullptr);
}

// The native handle does not change over the lifetime of the queue.
TEST_P(olGetQueueInfoTest, NativeHandleStable) {
  void *First = nullptr;
  void *Second = nullptr;
  ASSERT_SUCCESS_OR_UNSUPPORTED(olGetQueueInfo(
      Queue, OL_QUEUE_INFO_NATIVE_HANDLE, sizeof(First), &First));
  ASSERT_SUCCESS_OR_UNSUPPORTED(olGetQueueInfo(
      Queue, OL_QUEUE_INFO_NATIVE_HANDLE, sizeof(Second), &Second));
  ASSERT_NE(First, nullptr);
  ASSERT_EQ(First, Second);
}

// The size reported by olGetQueueInfoSize is accepted by olGetQueueInfo and
// one byte less is rejected, for every property.
TEST_P(olGetQueueInfoTest, SizeRoundTrip) {
  for (ol_queue_info_t Prop :
       {OL_QUEUE_INFO_DEVICE, OL_QUEUE_INFO_CONTEXT, OL_QUEUE_INFO_EMPTY}) {
    SCOPED_TRACE(Prop);
    size_t Size = 0;
    ASSERT_SUCCESS(olGetQueueInfoSize(Queue, Prop, &Size));
    ASSERT_GT(Size, 0u);

    std::vector<char> Buffer(Size);
    ASSERT_SUCCESS(olGetQueueInfo(Queue, Prop, Size, Buffer.data()));
    ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
                 olGetQueueInfo(Queue, Prop, Size - 1, Buffer.data()));
  }

  // NATIVE_HANDLE is checked last because a platform may not support it, which
  // skips the test.
  size_t Size = 0;
  ASSERT_SUCCESS_OR_UNSUPPORTED(
      olGetQueueInfoSize(Queue, OL_QUEUE_INFO_NATIVE_HANDLE, &Size));
  ASSERT_GT(Size, 0u);
  std::vector<char> Buffer(Size);
  ASSERT_SUCCESS_OR_UNSUPPORTED(
      olGetQueueInfo(Queue, OL_QUEUE_INFO_NATIVE_HANDLE, Size, Buffer.data()));
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetQueueInfo(Queue, OL_QUEUE_INFO_NATIVE_HANDLE, Size - 1,
                              Buffer.data()));
}

// olGetQueueInfo is thread safe; concurrent queries return consistent values.
TEST_P(olGetQueueInfoTest, ConcurrentQueries) {
  constexpr size_t NumThreads = 4;
  struct Result {
    ol_result_t DeviceRes = nullptr;
    ol_result_t ContextRes = nullptr;
    ol_device_handle_t RetrievedDevice = nullptr;
    ol_context_handle_t RetrievedContext = nullptr;
  };
  std::vector<Result> Results(NumThreads);
  std::vector<std::thread> Threads;
  for (size_t I = 0; I < NumThreads; ++I)
    Threads.emplace_back([&, I] {
      Result &R = Results[I];
      for (int Iter = 0; Iter < 100; ++Iter) {
        R.DeviceRes =
            olGetQueueInfo(Queue, OL_QUEUE_INFO_DEVICE,
                           sizeof(R.RetrievedDevice), &R.RetrievedDevice);
        if (R.DeviceRes)
          return;
        R.ContextRes =
            olGetQueueInfo(Queue, OL_QUEUE_INFO_CONTEXT,
                           sizeof(R.RetrievedContext), &R.RetrievedContext);
        if (R.ContextRes)
          return;
      }
    });
  for (std::thread &T : Threads)
    T.join();

  for (const Result &R : Results) {
    ASSERT_SUCCESS(R.DeviceRes);
    ASSERT_SUCCESS(R.ContextRes);
    ASSERT_EQ(R.RetrievedDevice, Device);
    ASSERT_EQ(R.RetrievedContext, Context);
  }
}
