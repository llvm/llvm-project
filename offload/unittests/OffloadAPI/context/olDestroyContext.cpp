//===------- Offload API tests - olDestroyContext -------------------------===//
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
#include <mutex>
#include <thread>
#include <vector>

using olDestroyContextTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olDestroyContextTest);

TEST_P(olDestroyContextTest, Success) {
  ol_context_handle_t Context = nullptr;
  ASSERT_SUCCESS(olCreateContext(1, &Device, &Context));
  ASSERT_SUCCESS(olDestroyContext(Context));
}

TEST_P(olDestroyContextTest, InvalidNullHandle) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olDestroyContext(nullptr));
}

// Destroying a context does not affect its devices: new contexts and queues can
// be created on them afterwards.
TEST_P(olDestroyContextTest, SuccessDeviceUsableAfterwards) {
  ASSERT_SUCCESS(olDestroyContext(Context));
  Context = nullptr;

  ol_context_handle_t NewContext = nullptr;
  ASSERT_SUCCESS(olCreateContext(1, &Device, &NewContext));
  ol_queue_handle_t Queue = nullptr;
  ASSERT_SUCCESS(olCreateQueue(NewContext, Device, &Queue));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_SUCCESS(olDestroyQueue(Queue));
  ASSERT_SUCCESS(olDestroyContext(NewContext));
}

// olDestroyContext blocks until work pending on queues that were already
// destroyed has finished.
TEST_P(olDestroyContextTest, SuccessWaitsForPendingWork) {
  SKIP_IF_FORCE_SYNC_OPS();

  ol_queue_handle_t Queue = nullptr;
  ASSERT_SUCCESS(olCreateQueue(Context, Device, &Queue));
  ManuallyTriggeredTask Task;
  ASSERT_SUCCESS(olLaunchHostFunction(
      Queue,
      [](void *Data) { static_cast<ManuallyTriggeredTask *>(Data)->wait(); },
      &Task));
  ASSERT_SUCCESS(olDestroyQueue(Queue));

  // The sleep only makes the blocking observable; the assertion on Released
  // holds for any delay.
  std::atomic<bool> Released{false};
  std::thread Releaser([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::lock_guard<std::mutex> Lock(Task.M);
    Released = true;
    Task.Flag = true;
    Task.CV.notify_one();
  });
  ol_result_t Status = olDestroyContext(Context);
  bool ReleasedBeforeReturn = Released.load();
  Context = nullptr;
  Releaser.join();

  ASSERT_SUCCESS(Status);
  ASSERT_TRUE(ReleasedBeforeReturn);
}

// olDestroyContext may be called concurrently on distinct contexts.
TEST_P(olDestroyContextTest, ConcurrentDestroyDistinctContexts) {
  constexpr size_t NumThreads = 4;
  struct Result {
    ol_result_t CreateRes = nullptr;
    ol_result_t DestroyRes = nullptr;
    ol_context_handle_t Context = nullptr;
  };
  std::vector<Result> Results(NumThreads);
  for (Result &R : Results)
    R.CreateRes = olCreateContext(1, &Device, &R.Context);
  for (const Result &R : Results)
    ASSERT_SUCCESS(R.CreateRes);

  std::vector<std::thread> Threads;
  for (size_t I = 0; I < NumThreads; ++I)
    Threads.emplace_back([&, I] {
      Results[I].DestroyRes = olDestroyContext(Results[I].Context);
    });
  for (std::thread &T : Threads)
    T.join();

  for (const Result &R : Results)
    ASSERT_SUCCESS(R.DestroyRes);
}
