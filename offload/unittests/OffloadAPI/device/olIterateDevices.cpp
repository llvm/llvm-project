//===------- Offload API tests - olIterateDevices -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olIterateDevicesTest = OffloadTest;

TEST_F(olIterateDevicesTest, SuccessEmptyCallback) {
  ASSERT_SUCCESS(olIterateDevices(
      [](ol_device_handle_t, void *) { return false; }, nullptr));
}

TEST_F(olIterateDevicesTest, SuccessGetDevice) {
  uint32_t DeviceCount = 0;
  ol_device_handle_t Device = nullptr;

  ASSERT_SUCCESS(olIterateDevices(
      [](ol_device_handle_t, void *Data) {
        auto Count = static_cast<uint32_t *>(Data);
        *Count += 1;
        return false;
      },
      &DeviceCount));

  if (DeviceCount == 0) {
    GTEST_SKIP() << "No available devices.";
  }

  ASSERT_SUCCESS(olIterateDevices(
      [](ol_device_handle_t D, void *Data) {
        auto DevicePtr = static_cast<ol_device_handle_t *>(Data);
        *DevicePtr = D;
        return true;
      },
      &Device));

  ASSERT_NE(Device, nullptr);
}
