//===------- Offload API tests - offloadDeviceGetCount --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/fixtures.hpp"
#include <gtest/gtest.h>
#include <offload_api.h>

using offloadDeviceGetCountTest = offloadPlatformTest;

TEST_F(offloadDeviceGetCountTest, Success) {
  uint32_t Count = 0;
  ASSERT_SUCCESS(offloadDeviceGetCount(Platform, &Count));
  ASSERT_NE(Count, 0lu);
}

TEST_F(offloadDeviceGetCountTest, InvalidNullPlatform) {
  uint32_t Count = 0;
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_NULL_HANDLE,
               offloadDeviceGetCount(nullptr, &Count));
}

TEST_F(offloadDeviceGetCountTest, InvalidNullPointer) {
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_NULL_POINTER,
               offloadDeviceGetCount(Platform, nullptr));
}
