//===------- Offload API tests - olGetDeviceCount --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olGetDeviceCountTest = offloadPlatformTest;

TEST_F(olGetDeviceCountTest, Success) {
  uint32_t Count = 0;
  ASSERT_SUCCESS(olGetDeviceCount(Platform, &Count));
}

TEST_F(olGetDeviceCountTest, InvalidNullPlatform) {
  uint32_t Count = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olGetDeviceCount(nullptr, &Count));
}

TEST_F(olGetDeviceCountTest, InvalidNullPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetDeviceCount(Platform, nullptr));
}
