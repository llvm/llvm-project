//===------- Offload API tests - olGetPlatformCount ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olGetPlatformCountTest = offloadTest;

TEST_F(olGetPlatformCountTest, Success) {
  uint32_t PlatformCount;
  ASSERT_SUCCESS(olGetPlatformCount(&PlatformCount));
}

TEST_F(olGetPlatformCountTest, InvalidNullPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER, olGetPlatformCount(nullptr));
}
