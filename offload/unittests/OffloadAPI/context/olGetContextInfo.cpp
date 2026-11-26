//===------- Offload API tests - olGetContextInfo -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <algorithm>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using olGetContextInfoTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetContextInfoTest);

TEST_P(olGetContextInfoTest, SuccessNumDevices) {
  size_t NumDevices = 0;
  ASSERT_SUCCESS(olGetContextInfo(Context, OL_CONTEXT_INFO_NUM_DEVICES,
                                  sizeof(NumDevices), &NumDevices));
  ASSERT_EQ(NumDevices, 1u);
}

TEST_P(olGetContextInfoTest, SuccessDevices) {
  ol_device_handle_t OutDevice = nullptr;
  ASSERT_SUCCESS(olGetContextInfo(Context, OL_CONTEXT_INFO_DEVICES,
                                  sizeof(OutDevice), &OutDevice));
  ASSERT_EQ(OutDevice, Device);
}

TEST_P(olGetContextInfoTest, SuccessPlatform) {
  ol_platform_handle_t ContextPlatform = nullptr;
  ASSERT_SUCCESS(olGetContextInfo(Context, OL_CONTEXT_INFO_PLATFORM,
                                  sizeof(ContextPlatform), &ContextPlatform));

  ol_platform_handle_t DevicePlatform = nullptr;
  ASSERT_SUCCESS(olGetDeviceInfo(Device, OL_DEVICE_INFO_PLATFORM,
                                 sizeof(DevicePlatform), &DevicePlatform));
  ASSERT_EQ(ContextPlatform, DevicePlatform);
}

TEST_P(olGetContextInfoTest, InvalidNullHandle) {
  size_t NumDevices = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetContextInfo(nullptr, OL_CONTEXT_INFO_NUM_DEVICES,
                                sizeof(NumDevices), &NumDevices));
}

TEST_P(olGetContextInfoTest, InvalidSizeSmall) {
  size_t NumDevices = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetContextInfo(Context, OL_CONTEXT_INFO_NUM_DEVICES,
                                sizeof(NumDevices) - 1, &NumDevices));
}

TEST_P(olGetContextInfoTest, SuccessNativeHandle) {
  void *Handle = nullptr;
  ASSERT_SUCCESS_OR_UNSUPPORTED(olGetContextInfo(
      Context, OL_CONTEXT_INFO_NATIVE_HANDLE, sizeof(Handle), &Handle));
  ASSERT_NE(Handle, nullptr);
}

TEST_P(olGetContextInfoTest, InvalidSizeZero) {
  size_t NumDevices = 0;
  ASSERT_ERROR(
      OL_ERRC_INVALID_SIZE,
      olGetContextInfo(Context, OL_CONTEXT_INFO_NUM_DEVICES, 0, &NumDevices));
}

TEST_P(olGetContextInfoTest, InvalidNullPointerPropValue) {
  size_t NumDevices = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetContextInfo(Context, OL_CONTEXT_INFO_NUM_DEVICES,
                                sizeof(NumDevices), nullptr));
}

TEST_P(olGetContextInfoTest, InvalidEnumeration) {
  size_t Value = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetContextInfo(Context, OL_CONTEXT_INFO_FORCE_UINT32,
                                sizeof(Value), &Value));
}

// A PropSize smaller than the DEVICES array is rejected.
TEST_P(olGetContextInfoTest, InvalidSizeSmallDevices) {
  ol_device_handle_t OutDevice = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetContextInfo(Context, OL_CONTEXT_INFO_DEVICES,
                                sizeof(OutDevice) - 1, &OutDevice));
}

// The size reported by olGetContextInfoSize and any larger size are accepted by
// olGetContextInfo and one byte less is rejected.
TEST_P(olGetContextInfoTest, SizeRoundTrip) {
  for (ol_context_info_t Prop :
       {OL_CONTEXT_INFO_NUM_DEVICES, OL_CONTEXT_INFO_DEVICES,
        OL_CONTEXT_INFO_PLATFORM}) {
    SCOPED_TRACE(Prop);
    size_t Size = 0;
    ASSERT_SUCCESS(olGetContextInfoSize(Context, Prop, &Size));
    ASSERT_GT(Size, 0u);

    std::vector<char> Buffer(Size + 1);
    ASSERT_SUCCESS(olGetContextInfo(Context, Prop, Size, Buffer.data()));
    ASSERT_SUCCESS(olGetContextInfo(Context, Prop, Size + 1, Buffer.data()));
    ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
                 olGetContextInfo(Context, Prop, Size - 1, Buffer.data()));
  }

  // NATIVE_HANDLE is checked last because a platform may not support it, which
  // skips the test.
  size_t Size = 0;
  ASSERT_SUCCESS_OR_UNSUPPORTED(
      olGetContextInfoSize(Context, OL_CONTEXT_INFO_NATIVE_HANDLE, &Size));
  ASSERT_GT(Size, 0u);
  std::vector<char> Buffer(Size);
  ASSERT_SUCCESS_OR_UNSUPPORTED(olGetContextInfo(
      Context, OL_CONTEXT_INFO_NATIVE_HANDLE, Size, Buffer.data()));
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetContextInfo(Context, OL_CONTEXT_INFO_NATIVE_HANDLE,
                                Size - 1, Buffer.data()));
}

// The native handle stays the same for the lifetime of the context, if the
// platform supports it.
TEST_P(olGetContextInfoTest, NativeHandleStable) {
  void *First = nullptr;
  void *Second = nullptr;
  ASSERT_SUCCESS_OR_UNSUPPORTED(olGetContextInfo(
      Context, OL_CONTEXT_INFO_NATIVE_HANDLE, sizeof(First), &First));
  ASSERT_SUCCESS(olGetContextInfo(Context, OL_CONTEXT_INFO_NATIVE_HANDLE,
                                  sizeof(Second), &Second));
  ASSERT_EQ(First, Second);
}

// The host platform has no native context handle; the query reports
// UNSUPPORTED instead of failing.
TEST_F(OffloadTest, olGetContextInfoNativeHandleUnsupportedOnHost) {
  ol_context_handle_t HostContext = nullptr;
  ASSERT_SUCCESS(olCreateContext(1, &Host, &HostContext));
  void *Handle = nullptr;
  ol_result_t Status = olGetContextInfo(
      HostContext, OL_CONTEXT_INFO_NATIVE_HANDLE, sizeof(Handle), &Handle);
  ASSERT_SUCCESS(olDestroyContext(HostContext));
  ASSERT_ERROR(OL_ERRC_UNSUPPORTED, Status);
}

// olGetContextInfo is thread safe; concurrent queries return consistent values.
TEST_P(olGetContextInfoTest, ConcurrentQueries) {
  constexpr size_t NumThreads = 4;
  struct Result {
    ol_result_t NumRes = nullptr;
    ol_result_t PlatformRes = nullptr;
    size_t NumDevices = 0;
    ol_platform_handle_t Platform = nullptr;
    ol_result_t DevicesRes = nullptr;
    ol_device_handle_t OutDevice = nullptr;
  };
  std::vector<Result> Results(NumThreads);
  std::vector<std::thread> Threads;
  for (size_t I = 0; I < NumThreads; ++I)
    Threads.emplace_back([&, I] {
      Result &R = Results[I];
      for (int Iter = 0; Iter < 100; ++Iter) {
        R.NumRes = olGetContextInfo(Context, OL_CONTEXT_INFO_NUM_DEVICES,
                                    sizeof(R.NumDevices), &R.NumDevices);
        if (R.NumRes)
          return;
        R.PlatformRes = olGetContextInfo(Context, OL_CONTEXT_INFO_PLATFORM,
                                         sizeof(R.Platform), &R.Platform);
        if (R.PlatformRes)
          return;
        R.DevicesRes = olGetContextInfo(Context, OL_CONTEXT_INFO_DEVICES,
                                        sizeof(R.OutDevice), &R.OutDevice);
        if (R.DevicesRes)
          return;
      }
    });
  for (std::thread &T : Threads)
    T.join();

  ol_platform_handle_t Expected = nullptr;
  ASSERT_SUCCESS(olGetDeviceInfo(Device, OL_DEVICE_INFO_PLATFORM,
                                 sizeof(Expected), &Expected));
  for (const Result &R : Results) {
    ASSERT_SUCCESS(R.NumRes);
    ASSERT_SUCCESS(R.PlatformRes);
    ASSERT_SUCCESS(R.DevicesRes);
    ASSERT_EQ(R.OutDevice, Device);
    ASSERT_EQ(R.NumDevices, 1u);
    ASSERT_EQ(R.Platform, Expected);
  }
}

// DEVICES lists the devices given to olCreateContext, NUM_DEVICES counts them,
// and PLATFORM is the platform they all belong to.
using olGetContextInfoMultiDeviceTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_MULTI_DEVICE_FIXTURE(olGetContextInfoMultiDeviceTest);

TEST_P(olGetContextInfoMultiDeviceTest, SuccessDevices) {
  size_t NumDevices = 0;
  ASSERT_SUCCESS(olGetContextInfo(Context, OL_CONTEXT_INFO_NUM_DEVICES,
                                  sizeof(NumDevices), &NumDevices));
  ASSERT_EQ(NumDevices, Devices.size());

  std::vector<ol_device_handle_t> Retrieved(NumDevices);
  ASSERT_SUCCESS(olGetContextInfo(Context, OL_CONTEXT_INFO_DEVICES,
                                  Retrieved.size() * sizeof(Retrieved[0]),
                                  Retrieved.data()));
  ASSERT_TRUE(std::is_permutation(Retrieved.begin(), Retrieved.end(),
                                  Devices.begin(), Devices.end()));
}

TEST_P(olGetContextInfoMultiDeviceTest, SuccessPlatformMatchesDevices) {
  ol_platform_handle_t ContextPlatform = nullptr;
  ASSERT_SUCCESS(olGetContextInfo(Context, OL_CONTEXT_INFO_PLATFORM,
                                  sizeof(ContextPlatform), &ContextPlatform));
  for (ol_device_handle_t D : Devices) {
    ol_platform_handle_t DevicePlatform = nullptr;
    ASSERT_SUCCESS(olGetDeviceInfo(D, OL_DEVICE_INFO_PLATFORM,
                                   sizeof(DevicePlatform), &DevicePlatform));
    ASSERT_EQ(ContextPlatform, DevicePlatform);
  }
}
