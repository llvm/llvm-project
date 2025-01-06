//===------- Offload API tests - olGetPlatform -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olGetPlatformTest = offloadTest;

TEST_F(olGetPlatformTest, Success) {
  uint32_t PlatformCount;
  ASSERT_SUCCESS(olGetPlatformCount(&PlatformCount));
  std::vector<ol_platform_handle_t> Platforms(PlatformCount);
  ASSERT_SUCCESS(olGetPlatform(PlatformCount, Platforms.data()));
}

TEST_F(olGetPlatformTest, InvalidNumEntries) {
  uint32_t PlatformCount;
  ASSERT_SUCCESS(olGetPlatformCount(&PlatformCount));
  std::vector<ol_platform_handle_t> Platforms(PlatformCount);
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetPlatform(PlatformCount + 1, Platforms.data()));
}
