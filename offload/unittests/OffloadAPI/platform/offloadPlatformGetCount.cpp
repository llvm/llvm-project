//===------- Offload API tests - offloadPlatformGetCount ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/fixtures.hpp"
#include <gtest/gtest.h>
#include <offload_api.h>

using offloadPlatformGetCountTest = offloadTest;

TEST_F(offloadPlatformGetCountTest, Success) {
  uint32_t PlatformCount;
  ASSERT_SUCCESS(offloadPlatformGetCount(&PlatformCount));
}

TEST_F(offloadPlatformGetCountTest, InvalidNullPointer) {
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_NULL_POINTER,
               offloadPlatformGetCount(nullptr));
}
