//===------- Offload API tests - olGetPlatformInfo -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Properties.hpp"

using olGetPlatformInfoTest = OffloadPlatformTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetPlatformInfoTest);

using olGetPlatformInfoNamesTest =
    OffloadPlatformTestWithParam<ol_platform_info_t>;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE_WITH_PARAM(
    olGetPlatformInfoNamesTest, PlatformInfoNames,
    defaultPrinterWithParam<ol_platform_info_t>);

TEST_P(olGetPlatformInfoNamesTest, SuccessName) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetPlatformInfoSize(Platform, getTestParam(), &Size));
  ASSERT_GT(Size, 0ul);
  std::vector<char> Name;
  Name.resize(Size);
  ASSERT_SUCCESS(
      olGetPlatformInfo(Platform, getTestParam(), Size, Name.data()));
  ASSERT_EQ(std::strlen(Name.data()), Size - 1);
}

TEST_P(olGetPlatformInfoTest, SuccessBackend) {
  ol_platform_backend_t Backend;
  ASSERT_SUCCESS(olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND,
                                   sizeof(ol_platform_backend_t), &Backend));
}

TEST_P(olGetPlatformInfoTest, InvalidNullHandle) {
  ol_platform_backend_t Backend;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetPlatformInfo(nullptr, OL_PLATFORM_INFO_BACKEND,
                                 sizeof(Backend), &Backend));
}

TEST_P(olGetPlatformInfoTest, InvalidPlatformInfoEnumeration) {
  ol_platform_backend_t Backend;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetPlatformInfo(Platform, OL_PLATFORM_INFO_FORCE_UINT32,
                                 sizeof(Backend), &Backend));
}

TEST_P(olGetPlatformInfoTest, InvalidSizeZero) {
  ol_platform_backend_t Backend;
  ASSERT_ERROR(
      OL_ERRC_INVALID_SIZE,
      olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND, 0, &Backend));
}

TEST_P(olGetPlatformInfoTest, InvalidSizeSmall) {
  ol_platform_backend_t Backend;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND,
                                 sizeof(Backend) - 1, &Backend));
}

TEST_P(olGetPlatformInfoTest, InvalidNullPointerPropValue) {
  ol_platform_backend_t Backend;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND,
                                 sizeof(Backend), nullptr));
}
