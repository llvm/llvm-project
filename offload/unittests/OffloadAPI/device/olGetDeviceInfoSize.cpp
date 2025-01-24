//===------- Offload API tests - olGetDeviceInfoSize -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"
#include "olDeviceInfo.hpp"

struct olGetDeviceInfoSizeTest
    : offloadDeviceTest,
      ::testing::WithParamInterface<ol_device_info_t> {

  void SetUp() override { RETURN_ON_FATAL_FAILURE(offloadDeviceTest::SetUp()); }
};

// TODO: We could autogenerate the list of enum values
INSTANTIATE_TEST_SUITE_P(
    , olGetDeviceInfoSizeTest, ::testing::ValuesIn(DeviceQueries),
    [](const ::testing::TestParamInfo<ol_device_info_t> &info) {
      std::stringstream ss;
      ss << info.param;
      return ss.str();
    });

TEST_P(olGetDeviceInfoSizeTest, Success) {
  ol_device_info_t InfoType = GetParam();
  size_t Size = 0;

  ASSERT_SUCCESS(olGetDeviceInfoSize(Device, InfoType, &Size));
  auto ExpectedSize = DeviceInfoSizeMap.find(InfoType);
  if (ExpectedSize != DeviceInfoSizeMap.end()) {
    ASSERT_EQ(Size, ExpectedSize->second);
  } else {
    ASSERT_NE(Size, 0lu);
  }
}

TEST_F(olGetDeviceInfoSizeTest, InvalidNullHandle) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetDeviceInfoSize(nullptr, OL_DEVICE_INFO_TYPE, &Size));
}

TEST_F(olGetDeviceInfoSizeTest, InvalidDeviceInfoEnumeration) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetDeviceInfoSize(Device, OL_DEVICE_INFO_FORCE_UINT32, &Size));
}

TEST_F(olGetDeviceInfoSizeTest, InvalidNullPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetDeviceInfoSize(Device, OL_DEVICE_INFO_TYPE, nullptr));
}
