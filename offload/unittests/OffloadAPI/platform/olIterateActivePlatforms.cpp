//===------- Offload API tests - olIterateActivePlatforms -----------------===//
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
#include <vector>

using olIterateActivePlatformsTest = OffloadTest;

TEST_F(olIterateActivePlatformsTest, SuccessEmptyCallback) {
  ASSERT_SUCCESS(olIterateActivePlatforms(
      [](ol_platform_handle_t, void *) { return false; }, nullptr));
}

TEST_F(olIterateActivePlatformsTest, SuccessActivePlatformsHaveDevices) {
  // Collect the platform of every device; this also initializes all platforms.
  std::vector<ol_platform_handle_t> DevicePlatforms;
  ASSERT_SUCCESS(olIterateDevices(
      [](ol_device_handle_t Device, void *Data) {
        ol_platform_handle_t Platform = nullptr;
        olGetDeviceInfo(Device, OL_DEVICE_INFO_PLATFORM, sizeof(Platform),
                        &Platform);
        static_cast<std::vector<ol_platform_handle_t> *>(Data)->push_back(
            Platform);
        return true;
      },
      &DevicePlatforms));

  std::vector<ol_platform_handle_t> ActivePlatforms;
  ASSERT_SUCCESS(olIterateActivePlatforms(
      [](ol_platform_handle_t Platform, void *Data) {
        static_cast<std::vector<ol_platform_handle_t> *>(Data)->push_back(
            Platform);
        return true;
      },
      &ActivePlatforms));

  // Every active platform must own at least one device, and every platform
  // owning a device is active once initialized.
  for (ol_platform_handle_t Platform : ActivePlatforms)
    ASSERT_NE(
        std::find(DevicePlatforms.begin(), DevicePlatforms.end(), Platform),
        DevicePlatforms.end());
  for (ol_platform_handle_t Platform : DevicePlatforms)
    ASSERT_NE(
        std::find(ActivePlatforms.begin(), ActivePlatforms.end(), Platform),
        ActivePlatforms.end());
}

TEST_F(olIterateActivePlatformsTest, SuccessSubsetOfAllPlatforms) {
  uint32_t AllCount = 0;
  ASSERT_SUCCESS(olIteratePlatforms(
      [](ol_platform_handle_t, void *Data) {
        ++*static_cast<uint32_t *>(Data);
        return true;
      },
      &AllCount));

  uint32_t ActiveCount = 0;
  ASSERT_SUCCESS(olIterateActivePlatforms(
      [](ol_platform_handle_t, void *Data) {
        ++*static_cast<uint32_t *>(Data);
        return true;
      },
      &ActiveCount));

  ASSERT_LE(ActiveCount, AllCount);
}
