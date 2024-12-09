//===------- Offload API tests - olGetDeviceInfo ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include "olDeviceInfo.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

struct olGetDeviceInfoTest : offloadDeviceTest,
                             ::testing::WithParamInterface<ol_device_info_t> {

  void SetUp() override { RETURN_ON_FATAL_FAILURE(offloadDeviceTest::SetUp()); }
};

INSTANTIATE_TEST_SUITE_P(
    , olGetDeviceInfoTest, ::testing::ValuesIn(DeviceQueries),
    [](const ::testing::TestParamInfo<ol_device_info_t> &info) {
      std::stringstream ss;
      ss << info.param;
      return ss.str();
    });

TEST_P(olGetDeviceInfoTest, Success) {
  ol_device_info_t InfoType = GetParam();
  size_t Size = 0;

  ASSERT_SUCCESS(olGetDeviceInfoSize(Device, InfoType, &Size));

  std::vector<char> InfoData(Size);
  ASSERT_SUCCESS(olGetDeviceInfo(Device, InfoType, Size, InfoData.data()));

  if (InfoType == OL_DEVICE_INFO_PLATFORM) {
    auto *ReturnedPlatform =
        reinterpret_cast<ol_platform_handle_t *>(InfoData.data());
    ASSERT_EQ(Platform, *ReturnedPlatform);
  }
}

TEST_F(olGetDeviceInfoTest, InvalidNullHandleDevice) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetDeviceInfo(nullptr, OL_DEVICE_INFO_TYPE,
                               sizeof(ol_device_type_t), &DeviceType));
}

TEST_F(olGetDeviceInfoTest, InvalidEnumerationInfoType) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetDeviceInfo(Device, OL_DEVICE_INFO_FORCE_UINT32,
                               sizeof(ol_device_type_t), &DeviceType));
}

TEST_F(olGetDeviceInfoTest, InvalidSizePropSize) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetDeviceInfo(Device, OL_DEVICE_INFO_TYPE, 0, &DeviceType));
}

TEST_F(olGetDeviceInfoTest, InvalidSizePropSizeSmall) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetDeviceInfo(Device, OL_DEVICE_INFO_TYPE,
                               sizeof(DeviceType) - 1, &DeviceType));
}

TEST_F(olGetDeviceInfoTest, InvalidNullPointerPropValue) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetDeviceInfo(Device, OL_DEVICE_INFO_TYPE, sizeof(DeviceType),
                               nullptr));
}
