//===------- Offload API tests - olGetQueueInfoSize -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"

#include <thread>
#include <vector>

using olGetQueueInfoSizeTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetQueueInfoSizeTest);

TEST_P(olGetQueueInfoSizeTest, SuccessDevice) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetQueueInfoSize(Queue, OL_QUEUE_INFO_DEVICE, &Size));
  ASSERT_EQ(Size, sizeof(ol_device_handle_t));
}

TEST_P(olGetQueueInfoSizeTest, SuccessContext) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetQueueInfoSize(Queue, OL_QUEUE_INFO_CONTEXT, &Size));
  ASSERT_EQ(Size, sizeof(ol_context_handle_t));
}

TEST_P(olGetQueueInfoSizeTest, SuccessEmpty) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetQueueInfoSize(Queue, OL_QUEUE_INFO_EMPTY, &Size));
  ASSERT_EQ(Size, sizeof(bool));
}

TEST_P(olGetQueueInfoSizeTest, InvalidNullHandle) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetQueueInfoSize(nullptr, OL_QUEUE_INFO_DEVICE, &Size));
}

TEST_P(olGetQueueInfoSizeTest, InvalidQueueInfoEnumeration) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetQueueInfoSize(Queue, OL_QUEUE_INFO_FORCE_UINT32, &Size));
}

TEST_P(olGetQueueInfoSizeTest, InvalidNullPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetQueueInfoSize(Queue, OL_QUEUE_INFO_DEVICE, nullptr));
}

TEST_P(olGetQueueInfoSizeTest, SuccessNativeHandle) {
  size_t Size = 0;
  ASSERT_SUCCESS_OR_UNSUPPORTED(
      olGetQueueInfoSize(Queue, OL_QUEUE_INFO_NATIVE_HANDLE, &Size));
  ASSERT_EQ(Size, sizeof(void *));
}

// olGetQueueInfoSize and olGetQueueInfo agree on whether NATIVE_HANDLE is
// supported.
TEST_P(olGetQueueInfoSizeTest, NativeHandleSupportAgreesWithInfo) {
  size_t Size = 0;
  ol_result_t SizeRes =
      olGetQueueInfoSize(Queue, OL_QUEUE_INFO_NATIVE_HANDLE, &Size);
  void *Handle = nullptr;
  ol_result_t InfoRes = olGetQueueInfo(Queue, OL_QUEUE_INFO_NATIVE_HANDLE,
                                       sizeof(Handle), &Handle);
  bool SizeUnsupported = SizeRes && SizeRes->Code == OL_ERRC_UNSUPPORTED;
  bool InfoUnsupported = InfoRes && InfoRes->Code == OL_ERRC_UNSUPPORTED;
  ASSERT_EQ(SizeUnsupported, InfoUnsupported);
  if (!SizeUnsupported) {
    ASSERT_SUCCESS(SizeRes);
    ASSERT_SUCCESS(InfoRes);
  }
}

// olGetQueueInfoSize is thread safe; concurrent queries return the same sizes.
TEST_P(olGetQueueInfoSizeTest, ConcurrentQueries) {
  constexpr size_t NumThreads = 4;
  struct Result {
    ol_result_t Res = nullptr;
    size_t DeviceSize = 0;
    size_t ContextSize = 0;
    size_t EmptySize = 0;
  };
  std::vector<Result> Results(NumThreads);
  std::vector<std::thread> Threads;
  for (size_t I = 0; I < NumThreads; ++I)
    Threads.emplace_back([&, I] {
      Result &R = Results[I];
      for (int Iter = 0; Iter < 100; ++Iter) {
        R.Res = olGetQueueInfoSize(Queue, OL_QUEUE_INFO_DEVICE, &R.DeviceSize);
        if (R.Res)
          return;
        R.Res =
            olGetQueueInfoSize(Queue, OL_QUEUE_INFO_CONTEXT, &R.ContextSize);
        if (R.Res)
          return;
        R.Res = olGetQueueInfoSize(Queue, OL_QUEUE_INFO_EMPTY, &R.EmptySize);
        if (R.Res)
          return;
      }
    });
  for (std::thread &T : Threads)
    T.join();

  for (const Result &R : Results) {
    ASSERT_SUCCESS(R.Res);
    ASSERT_EQ(R.DeviceSize, sizeof(ol_device_handle_t));
    ASSERT_EQ(R.ContextSize, sizeof(ol_context_handle_t));
    ASSERT_EQ(R.EmptySize, sizeof(bool));
  }
}
