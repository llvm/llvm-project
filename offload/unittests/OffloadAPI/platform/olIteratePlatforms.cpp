//===------- Offload API tests - olIteratePlatforms -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olIteratePlatformsTest = OffloadTest;

TEST_F(olIteratePlatformsTest, SuccessEmptyCallback) {
  ASSERT_SUCCESS(olIteratePlatforms(
      [](ol_platform_handle_t, void *) { return false; }, nullptr));
}

TEST_F(olIteratePlatformsTest, SuccessGetPlatform) {
  uint32_t PlatformCount = 0;
  ol_platform_handle_t Platform = nullptr;

  ASSERT_SUCCESS(olIteratePlatforms(
      [](ol_platform_handle_t, void *Data) {
        auto Count = static_cast<uint32_t *>(Data);
        *Count += 1;
        return true;
      },
      &PlatformCount));

  if (PlatformCount == 0) {
    GTEST_SKIP() << "No available platforms.";
  }

  ASSERT_SUCCESS(olIteratePlatforms(
      [](ol_platform_handle_t P, void *Data) {
        auto PlatformPtr = static_cast<ol_platform_handle_t *>(Data);
        *PlatformPtr = P;
        return true;
      },
      &Platform));

  ASSERT_NE(Platform, nullptr);
}
