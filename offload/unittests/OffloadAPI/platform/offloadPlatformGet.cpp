//===------- Offload API tests - offloadPlatformGet -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <offload_api.h>
#include "../common/fixtures.hpp"

using offloadPlatformGetTest = ::testing::Test;

TEST_F(offloadPlatformGetTest, Success) {
  uint32_t PlatformCount;
  ASSERT_SUCCESS(offloadPlatformGet(0, nullptr, &PlatformCount));
  std::vector<offload_platform_handle_t> Platforms(PlatformCount);
  ASSERT_SUCCESS(offloadPlatformGet(PlatformCount, Platforms.data(), nullptr));
}

TEST_F(offloadPlatformGetTest, InvalidNumEntries) {
  uint32_t PlatformCount;
  ASSERT_SUCCESS(offloadPlatformGet(0, nullptr, &PlatformCount));
  std::vector<offload_platform_handle_t> Platforms(PlatformCount);
  ASSERT_ERROR(
      OFFLOAD_ERRC_INVALID_SIZE,
      offloadPlatformGet(PlatformCount + 1, Platforms.data(), nullptr));
}
