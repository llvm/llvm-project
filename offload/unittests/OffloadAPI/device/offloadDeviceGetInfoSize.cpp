//===------- Offload API tests - offloadDeviceGetInfoSize -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <offload_api.h>

#include "../common/fixtures.hpp"
#include "offloadDeviceInfo.hpp"

struct offloadDeviceGetInfoSizeTest
    : offloadDeviceTest,
      ::testing::WithParamInterface<offload_device_info_t> {

  void SetUp() override { RETURN_ON_FATAL_FAILURE(offloadDeviceTest::SetUp()); }
};

// TODO: We could autogenerate the list of enum values
INSTANTIATE_TEST_SUITE_P(
    , offloadDeviceGetInfoSizeTest, ::testing::ValuesIn(DeviceQueries),
    [](const ::testing::TestParamInfo<offload_device_info_t> &info) {
      std::stringstream ss;
      ss << info.param;
      return ss.str();
    });

TEST_P(offloadDeviceGetInfoSizeTest, Success) {
  offload_device_info_t InfoType = GetParam();
  size_t Size = 0;

  ASSERT_SUCCESS(offloadDeviceGetInfoSize(Device, InfoType, &Size));
  auto ExpectedSize = DeviceInfoSizeMap.find(InfoType);
  if (ExpectedSize != DeviceInfoSizeMap.end()) {
    ASSERT_EQ(Size, ExpectedSize->second);
  } else {
    ASSERT_NE(Size, 0lu);
  }
}

TEST_F(offloadDeviceGetInfoSizeTest, InvalidNullHandle) {
  size_t Size = 0;
  ASSERT_ERROR(
      OFFLOAD_ERRC_INVALID_NULL_HANDLE,
      offloadDeviceGetInfoSize(nullptr, OFFLOAD_DEVICE_INFO_TYPE, &Size));
}

TEST_F(offloadDeviceGetInfoSizeTest, InvalidDeviceInfoEnumeration) {
  size_t Size = 0;
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_ENUMERATION,
               offloadDeviceGetInfoSize(
                   Device, OFFLOAD_DEVICE_INFO_FORCE_UINT32, &Size));
}

TEST_F(offloadDeviceGetInfoSizeTest, InvalidNullPointer) {
  ASSERT_ERROR(
      OFFLOAD_ERRC_INVALID_NULL_POINTER,
      offloadDeviceGetInfoSize(Device, OFFLOAD_DEVICE_INFO_TYPE, nullptr));
}
