//===------- Offload API tests - olGetPlatformInfoSize ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"
#include "olPlatformInfo.hpp"

struct olGetPlatformInfoSizeTest
    : offloadPlatformTest,
      ::testing::WithParamInterface<ol_platform_info_t> {};

INSTANTIATE_TEST_SUITE_P(
    olGetPlatformInfoSize, olGetPlatformInfoSizeTest,
    ::testing::ValuesIn(PlatformQueries),
    [](const ::testing::TestParamInfo<ol_platform_info_t> &info) {
      std::stringstream ss;
      ss << info.param;
      return ss.str();
    });

TEST_P(olGetPlatformInfoSizeTest, Success) {
  size_t Size = 0;
  ol_platform_info_t InfoType = GetParam();

  ASSERT_SUCCESS(olGetPlatformInfoSize(Platform, InfoType, &Size));
  auto ExpectedSize = PlatformInfoSizeMap.find(InfoType);
  if (ExpectedSize != PlatformInfoSizeMap.end()) {
    ASSERT_EQ(Size, ExpectedSize->second);
  } else {
    ASSERT_NE(Size, 0lu);
  }
}

TEST_F(olGetPlatformInfoSizeTest, InvalidNullHandle) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetPlatformInfoSize(nullptr, OL_PLATFORM_INFO_BACKEND, &Size));
}

TEST_F(olGetPlatformInfoSizeTest, InvalidPlatformInfoEnumeration) {
  size_t Size = 0;
  ASSERT_ERROR(
      OL_ERRC_INVALID_ENUMERATION,
      olGetPlatformInfoSize(Platform, OL_PLATFORM_INFO_FORCE_UINT32, &Size));
}

TEST_F(olGetPlatformInfoSizeTest, InvalidNullPointer) {
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_POINTER,
      olGetPlatformInfoSize(Platform, OL_PLATFORM_INFO_BACKEND, nullptr));
}
