//===------- Offload API tests - offloadDeviceGet -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/fixtures.hpp"
#include <gtest/gtest.h>
#include <offload_api.h>

using offloadDeviceGetTest = offloadPlatformTest;

TEST_F(offloadDeviceGetTest, Success) {
  uint32_t Count = 0;
  ASSERT_SUCCESS(
      offloadDeviceGet(Platform, OFFLOAD_DEVICE_TYPE_ALL, 0, nullptr, &Count));
  ASSERT_NE(Count, 0lu);
  std::vector<offload_device_handle_t> Devices(Count);
  ASSERT_SUCCESS(offloadDeviceGet(Platform, OFFLOAD_DEVICE_TYPE_ALL, Count,
                                  Devices.data(), nullptr));
  for (auto Device : Devices) {
    ASSERT_NE(nullptr, Device);
  }
}

TEST_F(offloadDeviceGetTest, SuccessSubsetOfDevices) {
  uint32_t Count;
  ASSERT_SUCCESS(
      offloadDeviceGet(Platform, OFFLOAD_DEVICE_TYPE_ALL, 0, nullptr, &Count));
  if (Count < 2) {
    GTEST_SKIP() << "Only one device is available on this platform.";
  }
  std::vector<offload_device_handle_t> Devices(Count - 1);
  ASSERT_SUCCESS(offloadDeviceGet(Platform, OFFLOAD_DEVICE_TYPE_ALL, Count - 1,
                                  Devices.data(), nullptr));
  for (auto Device : Devices) {
    ASSERT_NE(nullptr, Device);
  }
}
