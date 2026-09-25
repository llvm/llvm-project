//===------- Offload API tests - olIterateCompatiblePlatforms -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olIterateCompatiblePlatformsTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olIterateCompatiblePlatformsTest);

TEST_P(olIterateCompatiblePlatformsTest, SuccessEmptyCallback) {
  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Device, DeviceBin));

  ASSERT_SUCCESS(olIterateCompatiblePlatforms(
      DeviceBin->getBufferStart(), DeviceBin->getBufferSize(),
      [](ol_platform_handle_t, void *) { return false; }, nullptr));
}

TEST_P(olIterateCompatiblePlatformsTest, SuccessFindsOwnPlatform) {
  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Device, DeviceBin));

  ol_platform_handle_t OwnPlatform = nullptr;
  ASSERT_SUCCESS(olGetDeviceInfo(Device, OL_DEVICE_INFO_PLATFORM,
                                 sizeof(OwnPlatform), &OwnPlatform));

  bool Found = false;
  struct CallbackData {
    ol_platform_handle_t OwnPlatform;
    bool *Found;
  } Data{OwnPlatform, &Found};

  ASSERT_SUCCESS(olIterateCompatiblePlatforms(
      DeviceBin->getBufferStart(), DeviceBin->getBufferSize(),
      [](ol_platform_handle_t Platform, void *UserData) {
        auto *D = static_cast<CallbackData *>(UserData);
        if (Platform == D->OwnPlatform)
          *D->Found = true;
        return true;
      },
      &Data));

  ASSERT_TRUE(Found);
}

TEST_P(olIterateCompatiblePlatformsTest, InvalidBinary) {
  const char GarbageImage[] = "not a valid binary image";
  uint32_t Count = 0;
  ASSERT_SUCCESS(olIterateCompatiblePlatforms(
      GarbageImage, sizeof(GarbageImage),
      [](ol_platform_handle_t, void *Data) {
        *static_cast<uint32_t *>(Data) += 1;
        return true;
      },
      &Count));

  ASSERT_EQ(Count, 0u);
}

TEST_P(olIterateCompatiblePlatformsTest, NullPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olIterateCompatiblePlatforms(
                   nullptr, 42,
                   [](ol_platform_handle_t, void *) { return true; }, nullptr));
}
