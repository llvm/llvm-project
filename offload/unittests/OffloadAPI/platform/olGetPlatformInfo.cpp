//===------- Offload API tests - olGetPlatformInfo -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"

using olGetPlatformInfoTest = OffloadPlatformTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetPlatformInfoTest);

TEST_P(olGetPlatformInfoTest, SuccessName) {
  size_t Size = 0;
  EXPECT_SUCCESS(olGetPlatformInfoSize(Platform, OL_PLATFORM_INFO_NAME, &Size));
  EXPECT_GT(Size, 0ul);
  ASSERT_LT(Size, 0x1000);
  std::vector<char> Name;
  Name.resize(Size);
  EXPECT_SUCCESS(
      olGetPlatformInfo(Platform, OL_PLATFORM_INFO_NAME, Size, Name.data()));
  EXPECT_EQ(std::strlen(Name.data()), Size - 1);
}

TEST_P(olGetPlatformInfoTest, SuccessVendorName) {
  size_t Size = 0;
  EXPECT_SUCCESS(
      olGetPlatformInfoSize(Platform, OL_PLATFORM_INFO_VENDOR_NAME, &Size));
  EXPECT_GT(Size, 0ul);
  ASSERT_LT(Size, 0x1000);
  std::vector<char> VendorName;
  VendorName.resize(Size);
  EXPECT_SUCCESS(olGetPlatformInfo(Platform, OL_PLATFORM_INFO_VENDOR_NAME, Size,
                                   VendorName.data()));
  EXPECT_EQ(std::strlen(VendorName.data()), Size - 1);
}

TEST_P(olGetPlatformInfoTest, SuccessVersion) {
  size_t Size = 0;
  EXPECT_SUCCESS(
      olGetPlatformInfoSize(Platform, OL_PLATFORM_INFO_VERSION, &Size));
  EXPECT_GT(Size, 0ul);
  ASSERT_LT(Size, 0x1000);
  std::vector<char> Version;
  Version.resize(Size);
  EXPECT_SUCCESS(olGetPlatformInfo(Platform, OL_PLATFORM_INFO_VERSION, Size,
                                   Version.data()));
  EXPECT_EQ(std::strlen(Version.data()), Size - 1);
}

TEST_P(olGetPlatformInfoTest, SuccessBackend) {
  ol_platform_backend_t Backend;
  EXPECT_SUCCESS(olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND,
                                   sizeof(ol_platform_backend_t), &Backend));
}

TEST_P(olGetPlatformInfoTest, InvalidNullHandle) {
  ol_platform_backend_t Backend;
  EXPECT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetPlatformInfo(nullptr, OL_PLATFORM_INFO_BACKEND,
                                 sizeof(Backend), &Backend));
}

TEST_P(olGetPlatformInfoTest, InvalidPlatformInfoEnumeration) {
  ol_platform_backend_t Backend;
  EXPECT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetPlatformInfo(Platform, OL_PLATFORM_INFO_FORCE_UINT32,
                                 sizeof(Backend), &Backend));
}

TEST_P(olGetPlatformInfoTest, InvalidSizeZero) {
  ol_platform_backend_t Backend;
  EXPECT_ERROR(
      OL_ERRC_INVALID_SIZE,
      olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND, 0, &Backend));
}

TEST_P(olGetPlatformInfoTest, InvalidSizeSmall) {
  ol_platform_backend_t Backend;
  EXPECT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND,
                                 sizeof(Backend) - 1, &Backend));
}

TEST_P(olGetPlatformInfoTest, InvalidNullPointerPropValue) {
  ol_platform_backend_t Backend;
  EXPECT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND,
                                 sizeof(Backend), nullptr));
}
