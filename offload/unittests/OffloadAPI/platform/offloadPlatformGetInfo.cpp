//===------- Offload API tests - offloadPlatformGetInfo -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <offload_api.h>

#include "../common/fixtures.hpp"

struct offloadPlatformGetInfoTest
    : offloadPlatformTest,
      ::testing::WithParamInterface<offload_platform_info_t> {};

// TODO: We could autogenerate the list of enum values
INSTANTIATE_TEST_SUITE_P(
    offloadPlatformGetInfo, offloadPlatformGetInfoTest,
    ::testing::Values(OFFLOAD_PLATFORM_INFO_NAME,
                      OFFLOAD_PLATFORM_INFO_VENDOR_NAME,
                      OFFLOAD_PLATFORM_INFO_VERSION,
                      OFFLOAD_PLATFORM_INFO_BACKEND),
    [](const ::testing::TestParamInfo<offload_platform_info_t> &info) {
      std::stringstream ss;
      ss << info.param;
      return ss.str();
    });

// TODO: We could autogenerate this
std::unordered_map<offload_platform_info_t, size_t> PlatformInfoSizeMap = {
    {OFFLOAD_PLATFORM_INFO_BACKEND, sizeof(offload_platform_backend_t)},
};

TEST_P(offloadPlatformGetInfoTest, Success) {
  size_t Size = 0;
  offload_platform_info_t InfoType = GetParam();

  ASSERT_SUCCESS(offloadPlatformGetInfo(Platform, InfoType, 0, nullptr, &Size));
  auto ExpectedSize = PlatformInfoSizeMap.find(InfoType);
  if (ExpectedSize != PlatformInfoSizeMap.end()) {
    ASSERT_EQ(Size, ExpectedSize->second);
  } else {
    ASSERT_NE(Size, 0lu);
  }

  std::vector<char> InfoData(Size);
  ASSERT_SUCCESS(offloadPlatformGetInfo(Platform, InfoType, Size,
                                        InfoData.data(), nullptr));

  // Info types with a dynamic size are all char[] so we can verify the returned
  // string is the expected size.
  if (ExpectedSize == PlatformInfoSizeMap.end()) {
    ASSERT_EQ(Size, strlen(InfoData.data()) + 1);
  }
}

TEST_F(offloadPlatformGetInfoTest, InvalidNullHandle) {
  size_t Size = 0;
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_NULL_HANDLE,
               offloadPlatformGetInfo(nullptr, OFFLOAD_PLATFORM_INFO_BACKEND, 0,
                                      nullptr, &Size));
}

TEST_F(offloadPlatformGetInfoTest, InvalidPlatformInfoEnumeration) {
  size_t Size = 0;
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_ENUMERATION,
               offloadPlatformGetInfo(Platform,
                                      OFFLOAD_PLATFORM_INFO_FORCE_UINT32, 0,
                                      nullptr, &Size));
}

TEST_F(offloadPlatformGetInfoTest, InvalidSizeZero) {
  offload_platform_backend_t Backend;
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_SIZE,
               offloadPlatformGetInfo(Platform, OFFLOAD_PLATFORM_INFO_BACKEND,
                                      0, &Backend, nullptr));
}

TEST_F(offloadPlatformGetInfoTest, InvalidSizeSmall) {
  offload_platform_backend_t Backend;
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_SIZE,
               offloadPlatformGetInfo(Platform, OFFLOAD_PLATFORM_INFO_BACKEND,
                                      sizeof(Backend) - 1, &Backend, nullptr));
}

TEST_F(offloadPlatformGetInfoTest, InvalidNullPointerPropValue) {
  offload_platform_backend_t Backend;
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_NULL_POINTER,
               offloadPlatformGetInfo(Platform, OFFLOAD_PLATFORM_INFO_BACKEND,
                                      sizeof(Backend), nullptr, nullptr));
}

TEST_F(offloadPlatformGetInfoTest, InvalidNullPointerPropSizeRet) {
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_NULL_POINTER,
               offloadPlatformGetInfo(Platform, OFFLOAD_PLATFORM_INFO_BACKEND,
                                      0, nullptr, nullptr));
}
