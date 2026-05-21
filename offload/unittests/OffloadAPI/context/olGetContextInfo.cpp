//===------- Offload API tests - olGetContextInfo -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

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
