//===------- Offload API tests - olGetContextInfoSize ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using olGetContextInfoSizeTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetContextInfoSizeTest);

TEST_P(olGetContextInfoSizeTest, SuccessNumDevices) {
  size_t Size = 0;
  ASSERT_SUCCESS(
      olGetContextInfoSize(Context, OL_CONTEXT_INFO_NUM_DEVICES, &Size));
  ASSERT_EQ(Size, sizeof(size_t));
}

TEST_P(olGetContextInfoSizeTest, SuccessDevices) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetContextInfoSize(Context, OL_CONTEXT_INFO_DEVICES, &Size));
  ASSERT_EQ(Size, sizeof(ol_device_handle_t));
}

TEST_P(olGetContextInfoSizeTest, SuccessPlatform) {
  size_t Size = 0;
  ASSERT_SUCCESS(
      olGetContextInfoSize(Context, OL_CONTEXT_INFO_PLATFORM, &Size));
  ASSERT_EQ(Size, sizeof(ol_platform_handle_t));
}

TEST_P(olGetContextInfoSizeTest, InvalidNullHandle) {
  size_t Size = 0;
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_HANDLE,
      olGetContextInfoSize(nullptr, OL_CONTEXT_INFO_NUM_DEVICES, &Size));
}

TEST_P(olGetContextInfoSizeTest, InvalidNullOut) {
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_POINTER,
      olGetContextInfoSize(Context, OL_CONTEXT_INFO_NUM_DEVICES, nullptr));
}

TEST_P(olGetContextInfoSizeTest, SuccessNativeHandle) {
  size_t Size = 0;
  ASSERT_SUCCESS_OR_UNSUPPORTED(
      olGetContextInfoSize(Context, OL_CONTEXT_INFO_NATIVE_HANDLE, &Size));
  ASSERT_EQ(Size, sizeof(void *));
}

TEST_P(olGetContextInfoSizeTest, InvalidEnumeration) {
  size_t Size = 0;
  ASSERT_ERROR(
      OL_ERRC_INVALID_ENUMERATION,
      olGetContextInfoSize(Context, OL_CONTEXT_INFO_FORCE_UINT32, &Size));
}

// Whether a platform supports NATIVE_HANDLE is the same for both queries.
TEST_P(olGetContextInfoSizeTest, NativeHandleSupportAgreesWithInfo) {
  size_t Size = 0;
  ol_result_t SizeRes =
      olGetContextInfoSize(Context, OL_CONTEXT_INFO_NATIVE_HANDLE, &Size);
  void *Handle = nullptr;
  ol_result_t InfoRes = olGetContextInfo(Context, OL_CONTEXT_INFO_NATIVE_HANDLE,
                                         sizeof(Handle), &Handle);
  ASSERT_EQ(SizeRes == nullptr, InfoRes == nullptr);
  if (SizeRes) {
    ASSERT_EQ(SizeRes->Code, InfoRes->Code);
    ASSERT_ERROR(OL_ERRC_UNSUPPORTED, SizeRes);
  }
}

// olGetContextInfoSize is thread safe; concurrent queries return consistent
// values.
TEST_P(olGetContextInfoSizeTest, ConcurrentQueries) {
  constexpr size_t NumThreads = 4;
  struct Result {
    ol_result_t NumRes = nullptr;
    ol_result_t DevicesRes = nullptr;
    size_t NumSize = 0;
    size_t DevicesSize = 0;
  };
  std::vector<Result> Results(NumThreads);
  std::vector<std::thread> Threads;
  for (size_t I = 0; I < NumThreads; ++I)
    Threads.emplace_back([&, I] {
      Result &R = Results[I];
      for (int Iter = 0; Iter < 100; ++Iter) {
        R.NumRes = olGetContextInfoSize(Context, OL_CONTEXT_INFO_NUM_DEVICES,
                                        &R.NumSize);
        if (R.NumRes)
          return;
        R.DevicesRes = olGetContextInfoSize(Context, OL_CONTEXT_INFO_DEVICES,
                                            &R.DevicesSize);
        if (R.DevicesRes)
          return;
      }
    });
  for (std::thread &T : Threads)
    T.join();

  for (const Result &R : Results) {
    ASSERT_SUCCESS(R.NumRes);
    ASSERT_SUCCESS(R.DevicesRes);
    ASSERT_EQ(R.NumSize, sizeof(size_t));
    ASSERT_EQ(R.DevicesSize, sizeof(ol_device_handle_t));
  }
}

// DEVICES needs one handle per device of the context.
using olGetContextInfoSizeMultiDeviceTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_MULTI_DEVICE_FIXTURE(
    olGetContextInfoSizeMultiDeviceTest);

TEST_P(olGetContextInfoSizeMultiDeviceTest, SuccessDevices) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetContextInfoSize(Context, OL_CONTEXT_INFO_DEVICES, &Size));
  ASSERT_EQ(Size, Devices.size() * sizeof(ol_device_handle_t));
}
