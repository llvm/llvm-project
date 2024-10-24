//===------- Offload API tests - offloadDeviceGetInfo ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/fixtures.hpp"
#include <gtest/gtest.h>
#include <offload_api.h>

struct offloadDeviceGetInfoTest
    : offloadDeviceTest,
      ::testing::WithParamInterface<offload_device_info_t> {

  void SetUp() override { RETURN_ON_FATAL_FAILURE(offloadDeviceTest::SetUp()); }
};

// TODO: We could autogenerate the list of enum values
INSTANTIATE_TEST_SUITE_P(
    , offloadDeviceGetInfoTest,
    ::testing::Values(OFFLOAD_DEVICE_INFO_TYPE, OFFLOAD_DEVICE_INFO_PLATFORM,
                      OFFLOAD_DEVICE_INFO_NAME, OFFLOAD_DEVICE_INFO_VENDOR,
                      OFFLOAD_DEVICE_INFO_DRIVER_VERSION),
    [](const ::testing::TestParamInfo<offload_device_info_t> &info) {
      std::stringstream ss;
      ss << info.param;
      return ss.str();
    });

// TODO: We could autogenerate this
std::unordered_map<offload_device_info_t, size_t> DeviceInfoSizeMap = {
    {OFFLOAD_DEVICE_INFO_TYPE, sizeof(offload_device_type_t)},
    {OFFLOAD_DEVICE_INFO_PLATFORM, sizeof(offload_platform_handle_t)},
};

TEST_P(offloadDeviceGetInfoTest, Success) {
  offload_device_info_t InfoType = GetParam();
  size_t Size = 0;

  ASSERT_SUCCESS(offloadDeviceGetInfo(Device, InfoType, 0, nullptr, &Size));
  auto ExpectedSize = DeviceInfoSizeMap.find(InfoType);
  if (ExpectedSize != DeviceInfoSizeMap.end()) {
    ASSERT_EQ(Size, ExpectedSize->second);
  } else {
    ASSERT_NE(Size, 0lu);
  }

  std::vector<char> InfoData(Size);
  ASSERT_SUCCESS(
      offloadDeviceGetInfo(Device, InfoType, Size, InfoData.data(), nullptr));

  if (InfoType == OFFLOAD_DEVICE_INFO_PLATFORM) {
    auto *ReturnedPlatform =
        reinterpret_cast<offload_platform_handle_t *>(InfoData.data());
    ASSERT_EQ(Platform, *ReturnedPlatform);
  }
}

TEST_F(offloadDeviceGetInfoTest, InvalidNullHandleDevice) {
  offload_device_type_t DeviceType;
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_NULL_HANDLE,
               offloadDeviceGetInfo(nullptr, OFFLOAD_DEVICE_INFO_TYPE,
                                    sizeof(offload_device_type_t), &DeviceType,
                                    nullptr));
}

TEST_F(offloadDeviceGetInfoTest, InvalidEnumerationInfoType) {
  offload_device_type_t DeviceType;
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_ENUMERATION,
               offloadDeviceGetInfo(Device, OFFLOAD_DEVICE_INFO_FORCE_UINT32,
                                    sizeof(offload_device_type_t), &DeviceType,
                                    nullptr));
}

TEST_F(offloadDeviceGetInfoTest, InvalidSizePropSize) {
  offload_device_type_t DeviceType;
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_SIZE,
               offloadDeviceGetInfo(Device, OFFLOAD_DEVICE_INFO_TYPE, 0,
                                    &DeviceType, nullptr));
}

TEST_F(offloadDeviceGetInfoTest, InvalidSizePropSizeSmall) {
  offload_device_type_t DeviceType;
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_SIZE,
               offloadDeviceGetInfo(Device, OFFLOAD_DEVICE_INFO_TYPE,
                                    sizeof(DeviceType) - 1, &DeviceType,
                                    nullptr));
}

TEST_F(offloadDeviceGetInfoTest, InvalidNullPointerPropValue) {
  offload_device_type_t DeviceType;
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_NULL_POINTER,
               offloadDeviceGetInfo(Device, OFFLOAD_DEVICE_INFO_TYPE,
                                    sizeof(DeviceType), nullptr, nullptr));
}

TEST_F(offloadDeviceGetInfoTest, InvalidNullPointerPropSizeRet) {
  ASSERT_ERROR(OFFLOAD_ERRC_INVALID_NULL_POINTER,
               offloadDeviceGetInfo(Device, OFFLOAD_DEVICE_INFO_TYPE, 0,
                                    nullptr, nullptr));
}
