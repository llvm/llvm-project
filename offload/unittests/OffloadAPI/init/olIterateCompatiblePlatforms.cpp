//===------- Offload API tests - olIterateCompatiblePlatforms -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOTE: For this test suite, the implicit olInit/olShutDown doesn't happen, so
// tests have to do it themselves

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

struct olIterateCompatiblePlatformsBeforeInitTest : ::testing::Test {};

TEST_F(olIterateCompatiblePlatformsBeforeInitTest, SuccessBeforeInit) {
  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  if (!TestEnvironment::loadAnyDeviceBinary("foo", DeviceBin))
    GTEST_SKIP() << "No GPU device binary available for this build.";

  uint32_t Count = 0;
  ASSERT_SUCCESS(olIterateCompatiblePlatforms(
      DeviceBin->getBufferStart(), DeviceBin->getBufferSize(),
      [](ol_platform_handle_t Platform, void *Data) {
        EXPECT_NE(Platform, nullptr);
        *static_cast<uint32_t *>(Data) += 1;
        return true;
      },
      &Count));

  ASSERT_GT(Count, 0u);

  // olInit was never called, so nothing else should be usable yet.
  ASSERT_ERROR(OL_ERRC_UNINITIALIZED,
               olIterateDevices(
                   [](ol_device_handle_t, void *) { return false; }, nullptr));
}

TEST_F(olIterateCompatiblePlatformsBeforeInitTest, SuccessAfterInit) {
  ASSERT_SUCCESS(olInit(nullptr));

  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  if (!TestEnvironment::loadAnyDeviceBinary("foo", DeviceBin)) {
    ASSERT_SUCCESS(olShutDown());
    GTEST_SKIP() << "No GPU device binary available for this build.";
  }

  uint32_t Count = 0;
  ASSERT_SUCCESS(olIterateCompatiblePlatforms(
      DeviceBin->getBufferStart(), DeviceBin->getBufferSize(),
      [](ol_platform_handle_t Platform, void *Data) {
        EXPECT_NE(Platform, nullptr);
        *static_cast<uint32_t *>(Data) += 1;
        return true;
      },
      &Count));

  ASSERT_GT(Count, 0u);

  ASSERT_SUCCESS(olShutDown());
}
