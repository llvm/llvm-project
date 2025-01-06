//===------- Offload API tests - olGetPlatformInfo -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"
#include "olPlatformInfo.hpp"

struct olGetPlatformInfoTest
    : offloadPlatformTest,
      ::testing::WithParamInterface<ol_platform_info_t> {};

INSTANTIATE_TEST_SUITE_P(
    olGetPlatformInfo, olGetPlatformInfoTest,
    ::testing::ValuesIn(PlatformQueries),
    [](const ::testing::TestParamInfo<ol_platform_info_t> &info) {
      std::stringstream ss;
      ss << info.param;
      return ss.str();
    });

TEST_P(olGetPlatformInfoTest, Success) {
  size_t Size = 0;
  ol_platform_info_t InfoType = GetParam();

  ASSERT_SUCCESS(olGetPlatformInfoSize(Platform, InfoType, &Size));
  std::vector<char> InfoData(Size);
  ASSERT_SUCCESS(olGetPlatformInfo(Platform, InfoType, Size, InfoData.data()));

  // Info types with a dynamic size are all char[] so we can verify the returned
  // string is the expected size.
  auto ExpectedSize = PlatformInfoSizeMap.find(InfoType);
  if (ExpectedSize == PlatformInfoSizeMap.end()) {
    ASSERT_EQ(Size, strlen(InfoData.data()) + 1);
  }
}

TEST_F(olGetPlatformInfoTest, InvalidNullHandle) {
  ol_platform_backend_t Backend;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetPlatformInfo(nullptr, OL_PLATFORM_INFO_BACKEND,
                                 sizeof(Backend), &Backend));
}

TEST_F(olGetPlatformInfoTest, InvalidPlatformInfoEnumeration) {
  ol_platform_backend_t Backend;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetPlatformInfo(Platform, OL_PLATFORM_INFO_FORCE_UINT32,
                                 sizeof(Backend), &Backend));
}

TEST_F(olGetPlatformInfoTest, InvalidSizeZero) {
  ol_platform_backend_t Backend;
  ASSERT_ERROR(
      OL_ERRC_INVALID_SIZE,
      olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND, 0, &Backend));
}

TEST_F(olGetPlatformInfoTest, InvalidSizeSmall) {
  ol_platform_backend_t Backend;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND,
                                 sizeof(Backend) - 1, &Backend));
}

TEST_F(olGetPlatformInfoTest, InvalidNullPointerPropValue) {
  ol_platform_backend_t Backend;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND,
                                 sizeof(Backend), nullptr));
}
