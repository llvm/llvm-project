//===------- Offload API tests - olCreateContext --------------------------===//
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

// OffloadDeviceTest creates an ol_context_handle_t in SetUp; this fixture
// stops at the device so the tests below can exercise olCreateContext.
struct olCreateContextTest : OffloadTest,
                             ::testing::WithParamInterface<OffloadParam<int>> {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadTest::SetUp());
    auto &DevicesParam = std::get<0>(GetParam());
    if (!DevicesParam.empty())
      Device = DevicesParam.front().Handle;
    if (Device == nullptr)
      GTEST_SKIP() << "No available devices.";
  }

  ol_device_handle_t Device = nullptr;
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olCreateContextTest);

TEST_P(olCreateContextTest, Success) {
  ol_context_handle_t Context = nullptr;
  ASSERT_SUCCESS(olCreateContext(1, &Device, &Context));
  ASSERT_NE(Context, nullptr);
  ASSERT_SUCCESS(olDestroyContext(Context));
}

TEST_P(olCreateContextTest, InvalidNullDevices) {
  ol_context_handle_t Context = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olCreateContext(1, nullptr, &Context));
}

TEST_P(olCreateContextTest, InvalidZeroSize) {
  ol_context_handle_t Context = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE, olCreateContext(0, &Device, &Context));
}

TEST_P(olCreateContextTest, InvalidNullOut) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olCreateContext(1, &Device, nullptr));
}

// The devices passed to olCreateContext must belong to one platform: the host
// platform cannot be mixed with another one, whatever the order.
TEST_P(olCreateContextTest, InvalidDeviceAcrossPlatforms) {
  ol_context_handle_t Context = nullptr;
  ol_device_handle_t HostFirst[] = {Host, Device};
  ASSERT_ERROR(OL_ERRC_INVALID_DEVICE, olCreateContext(2, HostFirst, &Context));
  ol_device_handle_t HostLast[] = {Device, Host};
  ASSERT_ERROR(OL_ERRC_INVALID_DEVICE, olCreateContext(2, HostLast, &Context));
}

// The devices passed to olCreateContext must share one driver ID. This is only
// reachable with two drivers of the same platform, e.g. two Level Zero drivers.
TEST_F(OffloadTest, olCreateContextInvalidDeviceAcrossDriverIds) {
  const auto &All = TestEnvironment::getDevices();
  for (const auto &A : All)
    for (const auto &B : All) {
      ol_platform_handle_t PlatformA = nullptr, PlatformB = nullptr;
      uint32_t DriverA = 0, DriverB = 0;
      if (olGetDeviceInfo(A.Handle, OL_DEVICE_INFO_PLATFORM, sizeof(PlatformA),
                          &PlatformA) ||
          olGetDeviceInfo(B.Handle, OL_DEVICE_INFO_PLATFORM, sizeof(PlatformB),
                          &PlatformB) ||
          olGetDeviceInfo(A.Handle, OL_DEVICE_INFO_DRIVER_ID, sizeof(DriverA),
                          &DriverA) ||
          olGetDeviceInfo(B.Handle, OL_DEVICE_INFO_DRIVER_ID, sizeof(DriverB),
                          &DriverB))
        continue;
      if (PlatformA != PlatformB || DriverA == DriverB)
        continue;
      ol_device_handle_t Pair[] = {A.Handle, B.Handle};
      ol_context_handle_t Context = nullptr;
      ASSERT_ERROR(OL_ERRC_INVALID_DEVICE, olCreateContext(2, Pair, &Context));
      return;
    }
  GTEST_SKIP() << "No two devices of one platform with different driver IDs.";
}

// *Context is written only when olCreateContext succeeds.
TEST_P(olCreateContextTest, OutputUntouchedOnFailure) {
  ol_context_handle_t Sentinel = reinterpret_cast<ol_context_handle_t>(0x10);
  ol_context_handle_t Context = Sentinel;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE, olCreateContext(0, &Device, &Context));
  ASSERT_EQ(Context, Sentinel);

  ol_device_handle_t Mixed[] = {Host, Device};
  ASSERT_ERROR(OL_ERRC_INVALID_DEVICE, olCreateContext(2, Mixed, &Context));
  ASSERT_EQ(Context, Sentinel);
}

// olCreateContext copies Devices; the array may be freed or overwritten after
// the call.
TEST_P(olCreateContextTest, SuccessDevicesArrayReusable) {
  ol_device_handle_t Devices[] = {Device};
  ol_context_handle_t Context = nullptr;
  ASSERT_SUCCESS(olCreateContext(1, Devices, &Context));
  Devices[0] = nullptr;

  ol_device_handle_t Retrieved = nullptr;
  ASSERT_SUCCESS(olGetContextInfo(Context, OL_CONTEXT_INFO_DEVICES,
                                  sizeof(Retrieved), &Retrieved));
  ASSERT_EQ(Retrieved, Device);
  ASSERT_SUCCESS(olDestroyContext(Context));
}

// Several contexts may exist over the same device at the same time, and each
// can be destroyed independently.
TEST_P(olCreateContextTest, SuccessSeveralLiveContexts) {
  ol_context_handle_t First = nullptr, Second = nullptr;
  ASSERT_SUCCESS(olCreateContext(1, &Device, &First));
  ASSERT_SUCCESS(olCreateContext(1, &Device, &Second));
  ASSERT_NE(First, nullptr);
  ASSERT_NE(Second, nullptr);
  ASSERT_NE(First, Second);

  ASSERT_SUCCESS(olDestroyContext(First));
  ol_queue_handle_t Queue = nullptr;
  ASSERT_SUCCESS(olCreateQueue(Second, Device, &Queue));
  ASSERT_SUCCESS(olDestroyQueue(Queue));
  ASSERT_SUCCESS(olDestroyContext(Second));
}

// Contexts can be created and destroyed repeatedly.
TEST_P(olCreateContextTest, SuccessCreateDestroyLoop) {
  for (int I = 0; I < 50; ++I) {
    ol_context_handle_t Context = nullptr;
    ASSERT_SUCCESS(olCreateContext(1, &Device, &Context));
    ASSERT_SUCCESS(olDestroyContext(Context));
  }
}

// olCreateContext is thread safe: threads creating and destroying their own
// contexts over the same device do not interfere.
TEST_P(olCreateContextTest, ConcurrentCreateDestroy) {
  constexpr size_t NumThreads = 4;
  struct Result {
    ol_result_t CreateRes = nullptr;
    ol_result_t DestroyRes = nullptr;
    ol_context_handle_t Context = nullptr;
  };
  std::vector<Result> Results(NumThreads);
  std::vector<std::thread> Threads;
  for (size_t I = 0; I < NumThreads; ++I)
    Threads.emplace_back([&, I] {
      Result &R = Results[I];
      for (int Iter = 0; Iter < 20; ++Iter) {
        R.CreateRes = olCreateContext(1, &Device, &R.Context);
        if (R.CreateRes)
          return;
        R.DestroyRes = olDestroyContext(R.Context);
        if (R.DestroyRes)
          return;
      }
    });
  for (std::thread &T : Threads)
    T.join();

  for (const Result &R : Results) {
    ASSERT_SUCCESS(R.CreateRes);
    ASSERT_SUCCESS(R.DestroyRes);
  }
}

// Devices of one platform and driver can share a context.
using olCreateContextMultiDeviceTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_MULTI_DEVICE_FIXTURE(olCreateContextMultiDeviceTest);

TEST_P(olCreateContextMultiDeviceTest, SuccessMultipleDevices) {
  ol_context_handle_t Other = nullptr;
  ASSERT_SUCCESS(olCreateContext(Devices.size(), Devices.data(), &Other));
  ASSERT_NE(Other, nullptr);
  ASSERT_SUCCESS(olDestroyContext(Other));
}
