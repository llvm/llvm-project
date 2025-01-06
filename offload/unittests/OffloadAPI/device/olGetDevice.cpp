//===------- Offload API tests - olGetDevice -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olGetDeviceTest = offloadPlatformTest;

TEST_F(olGetDeviceTest, Success) {
  uint32_t Count = 0;
  ASSERT_SUCCESS(olGetDeviceCount(Platform, &Count));
  if (Count == 0)
    GTEST_SKIP() << "No available devices on this platform.";

  std::vector<ol_device_handle_t> Devices(Count);
  ASSERT_SUCCESS(olGetDevice(Platform, Count, Devices.data()));
  for (auto Device : Devices) {
    ASSERT_NE(nullptr, Device);
  }
}

TEST_F(olGetDeviceTest, SuccessSubsetOfDevices) {
  uint32_t Count;
  ASSERT_SUCCESS(olGetDeviceCount(Platform, &Count));
  if (Count < 2)
    GTEST_SKIP() << "Only one device is available on this platform.";

  std::vector<ol_device_handle_t> Devices(Count - 1);
  ASSERT_SUCCESS(olGetDevice(Platform, Count - 1, Devices.data()));
  for (auto Device : Devices) {
    ASSERT_NE(nullptr, Device);
  }
}
