//===------- Offload API tests - olGetPlatformInfoSize ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"

using olGetPlatformInfoSizeTest = OffloadPlatformTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetPlatformInfoSizeTest);

TEST_P(olGetPlatformInfoSizeTest, SuccessName) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetPlatformInfoSize(Platform, OL_PLATFORM_INFO_NAME, &Size));
  ASSERT_NE(Size, 0ul);
}

TEST_P(olGetPlatformInfoSizeTest, SuccessVendorName) {
  size_t Size = 0;
  ASSERT_SUCCESS(
      olGetPlatformInfoSize(Platform, OL_PLATFORM_INFO_VENDOR_NAME, &Size));
  ASSERT_NE(Size, 0ul);
}

TEST_P(olGetPlatformInfoSizeTest, SuccessVersion) {
  size_t Size = 0;
  ASSERT_SUCCESS(
      olGetPlatformInfoSize(Platform, OL_PLATFORM_INFO_VERSION, &Size));
  ASSERT_NE(Size, 0ul);
}

TEST_P(olGetPlatformInfoSizeTest, SuccessBackend) {
  size_t Size = 0;
  ASSERT_SUCCESS(
      olGetPlatformInfoSize(Platform, OL_PLATFORM_INFO_BACKEND, &Size));
  ASSERT_EQ(Size, sizeof(ol_platform_backend_t));
}

TEST_P(olGetPlatformInfoSizeTest, InvalidNullHandle) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetPlatformInfoSize(nullptr, OL_PLATFORM_INFO_BACKEND, &Size));
}

TEST_P(olGetPlatformInfoSizeTest, InvalidPlatformInfoEnumeration) {
  size_t Size = 0;
  ASSERT_ERROR(
      OL_ERRC_INVALID_ENUMERATION,
      olGetPlatformInfoSize(Platform, OL_PLATFORM_INFO_FORCE_UINT32, &Size));
}

TEST_P(olGetPlatformInfoSizeTest, InvalidNullPointer) {
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_POINTER,
      olGetPlatformInfoSize(Platform, OL_PLATFORM_INFO_BACKEND, nullptr));
}
