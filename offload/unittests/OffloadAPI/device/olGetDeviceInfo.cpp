//===------- Offload API tests - olGetDeviceInfo --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olGetDeviceInfoTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetDeviceInfoTest);

TEST_P(olGetDeviceInfoTest, SuccessType) {
  ol_device_type_t DeviceType;
  ASSERT_SUCCESS(olGetDeviceInfo(Device, OL_DEVICE_INFO_TYPE,
                                 sizeof(ol_device_type_t), &DeviceType));
}

TEST_P(olGetDeviceInfoTest, HostSuccessType) {
  ol_device_type_t DeviceType;
  ASSERT_SUCCESS(olGetDeviceInfo(Host, OL_DEVICE_INFO_TYPE,
                                 sizeof(ol_device_type_t), &DeviceType));
  ASSERT_EQ(DeviceType, OL_DEVICE_TYPE_HOST);
}

TEST_P(olGetDeviceInfoTest, SuccessPlatform) {
  ol_platform_handle_t Platform = nullptr;
  ASSERT_SUCCESS(olGetDeviceInfo(Device, OL_DEVICE_INFO_PLATFORM,
                                 sizeof(ol_platform_handle_t), &Platform));
  ASSERT_NE(Platform, nullptr);
}

TEST_P(olGetDeviceInfoTest, SuccessName) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(Device, OL_DEVICE_INFO_NAME, &Size));
  ASSERT_GT(Size, 0ul);
  std::vector<char> Name;
  Name.resize(Size);
  ASSERT_SUCCESS(
      olGetDeviceInfo(Device, OL_DEVICE_INFO_NAME, Size, Name.data()));
  ASSERT_EQ(std::strlen(Name.data()), Size - 1);
}

TEST_P(olGetDeviceInfoTest, HostName) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(Host, OL_DEVICE_INFO_NAME, &Size));
  ASSERT_GT(Size, 0ul);
  std::vector<char> Name;
  Name.resize(Size);
  ASSERT_SUCCESS(olGetDeviceInfo(Host, OL_DEVICE_INFO_NAME, Size, Name.data()));
  ASSERT_EQ(std::strlen(Name.data()), Size - 1);
}

TEST_P(olGetDeviceInfoTest, SuccessVendor) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(Device, OL_DEVICE_INFO_VENDOR, &Size));
  ASSERT_GT(Size, 0ul);
  std::vector<char> Vendor;
  Vendor.resize(Size);
  ASSERT_SUCCESS(
      olGetDeviceInfo(Device, OL_DEVICE_INFO_VENDOR, Size, Vendor.data()));
  ASSERT_EQ(std::strlen(Vendor.data()), Size - 1);
}

TEST_P(olGetDeviceInfoTest, SuccessDriverVersion) {
  size_t Size = 0;
  ASSERT_SUCCESS(
      olGetDeviceInfoSize(Device, OL_DEVICE_INFO_DRIVER_VERSION, &Size));
  ASSERT_GT(Size, 0ul);
  std::vector<char> DriverVersion;
  DriverVersion.resize(Size);
  ASSERT_SUCCESS(olGetDeviceInfo(Device, OL_DEVICE_INFO_DRIVER_VERSION, Size,
                                 DriverVersion.data()));
  ASSERT_EQ(std::strlen(DriverVersion.data()), Size - 1);
}

TEST_P(olGetDeviceInfoTest, SuccessMaxWorkGroupSize) {
  ol_dimensions_t Value{0, 0, 0};
  ASSERT_SUCCESS(olGetDeviceInfo(Device, OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE,
                                 sizeof(Value), &Value));
  ASSERT_GT(Value.x, 0u);
  ASSERT_GT(Value.y, 0u);
  ASSERT_GT(Value.z, 0u);
}

TEST_P(olGetDeviceInfoTest, InvalidNullHandleDevice) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetDeviceInfo(nullptr, OL_DEVICE_INFO_TYPE,
                               sizeof(ol_device_type_t), &DeviceType));
}

TEST_P(olGetDeviceInfoTest, InvalidEnumerationInfoType) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetDeviceInfo(Device, OL_DEVICE_INFO_FORCE_UINT32,
                               sizeof(ol_device_type_t), &DeviceType));
}

TEST_P(olGetDeviceInfoTest, InvalidSizePropSize) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetDeviceInfo(Device, OL_DEVICE_INFO_TYPE, 0, &DeviceType));
}

TEST_P(olGetDeviceInfoTest, InvalidSizePropSizeSmall) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetDeviceInfo(Device, OL_DEVICE_INFO_TYPE,
                               sizeof(DeviceType) - 1, &DeviceType));
}

TEST_P(olGetDeviceInfoTest, InvalidNullPointerPropValue) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetDeviceInfo(Device, OL_DEVICE_INFO_TYPE, sizeof(DeviceType),
                               nullptr));
}
