//===------- Offload API tests - offloadPlatformGetInfoSize ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <offload_api.h>

#include "../common/fixtures.hpp"
#include "offloadPlatformInfo.hpp"

struct offloadPlatformGetInfoSizeTest
    : offloadPlatformTest,
      ::testing::WithParamInterface<offload_platform_info_t> {};

INSTANTIATE_TEST_SUITE_P(
    offloadPlatformGetInfoSize, offloadPlatformGetInfoSizeTest,
    ::testing::ValuesIn(PlatformQueries),
    [](const ::testing::TestParamInfo<offload_platform_info_t> &info) {
      std::stringstream ss;
      ss << info.param;
      return ss.str();
    });

TEST_P(offloadPlatformGetInfoSizeTest, Success) {
  size_t Size = 0;
  offload_platform_info_t InfoType = GetParam();

  ASSERT_SUCCESS(offloadPlatformGetInfoSize(Platform, InfoType, &Size));
  auto ExpectedSize = PlatformInfoSizeMap.find(InfoType);
  if (ExpectedSize != PlatformInfoSizeMap.end()) {
    ASSERT_EQ(Size, ExpectedSize->second);
  } else {
    ASSERT_NE(Size, 0lu);
  }
}

TEST_F(offloadPlatformGetInfoSizeTest, InvalidNullHandle) {
  size_t Size = 0;
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_NULL_HANDLE,
               offloadPlatformGetInfoSize(
                   nullptr, OFFLOAD_PLATFORM_INFO_BACKEND, &Size));
}

TEST_F(offloadPlatformGetInfoSizeTest, InvalidPlatformInfoEnumeration) {
  size_t Size = 0;
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_ENUMERATION,
               offloadPlatformGetInfoSize(
                   Platform, OFFLOAD_PLATFORM_INFO_FORCE_UINT32, &Size));
}

TEST_F(offloadPlatformGetInfoSizeTest, InvalidNullPointer) {
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_NULL_POINTER,
               offloadPlatformGetInfoSize(
                   Platform, OFFLOAD_PLATFORM_INFO_BACKEND, nullptr));
}
