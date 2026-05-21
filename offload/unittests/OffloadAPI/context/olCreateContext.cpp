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

// OffloadDeviceTest creates an ol_context_handle_t in SetUp; this fixture
// stops at the device so the tests below can exercise olCreateContext.
struct olCreateContextTest
    : OffloadTest,
      ::testing::WithParamInterface<TestEnvironment::Device> {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadTest::SetUp());
    auto DeviceParam = GetParam();
    Device = DeviceParam.Handle;
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
