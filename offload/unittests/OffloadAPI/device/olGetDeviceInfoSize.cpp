//===------- Offload API tests - olGetDeviceInfoSize -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"

using olGetDeviceInfoSizeTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetDeviceInfoSizeTest);

TEST_P(olGetDeviceInfoSizeTest, SuccessType) {
  size_t Size = 0;
  EXPECT_SUCCESS(olGetDeviceInfoSize(Device, OL_DEVICE_INFO_TYPE, &Size));
  EXPECT_EQ(Size, sizeof(ol_device_type_t));
}

TEST_P(olGetDeviceInfoSizeTest, SuccessPlatform) {
  size_t Size = 0;
  EXPECT_SUCCESS(olGetDeviceInfoSize(Device, OL_DEVICE_INFO_PLATFORM, &Size));
  EXPECT_EQ(Size, sizeof(ol_platform_handle_t));
}

TEST_P(olGetDeviceInfoSizeTest, SuccessName) {
  size_t Size = 0;
  EXPECT_SUCCESS(olGetDeviceInfoSize(Device, OL_DEVICE_INFO_NAME, &Size));
  EXPECT_NE(Size, 0ul);
}

TEST_P(olGetDeviceInfoSizeTest, SuccessVendor) {
  size_t Size = 0;
  EXPECT_SUCCESS(olGetDeviceInfoSize(Device, OL_DEVICE_INFO_VENDOR, &Size));
  EXPECT_NE(Size, 0ul);
}

TEST_P(olGetDeviceInfoSizeTest, SuccessDriverVersion) {
  size_t Size = 0;
  EXPECT_SUCCESS(
      olGetDeviceInfoSize(Device, OL_DEVICE_INFO_DRIVER_VERSION, &Size));
  EXPECT_NE(Size, 0ul);
}

TEST_P(olGetDeviceInfoSizeTest, SuccessMaxWorkGroupSize) {
  size_t Size = 0;
  EXPECT_SUCCESS(
      olGetDeviceInfoSize(Device, OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE, &Size));
  EXPECT_EQ(Size, sizeof(uint32_t));
}

TEST_P(olGetDeviceInfoSizeTest, SuccessMaxWorkGroupSizePerDimension) {
  size_t Size = 0;
  EXPECT_SUCCESS(olGetDeviceInfoSize(
      Device, OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE_PER_DIMENSION, &Size));
  EXPECT_EQ(Size, sizeof(ol_dimensions_t));
  EXPECT_EQ(Size, sizeof(uint32_t) * 3);
}

TEST_P(olGetDeviceInfoSizeTest, InvalidNullHandle) {
  size_t Size = 0;
  EXPECT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetDeviceInfoSize(nullptr, OL_DEVICE_INFO_TYPE, &Size));
}

TEST_P(olGetDeviceInfoSizeTest, InvalidDeviceInfoEnumeration) {
  size_t Size = 0;
  EXPECT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetDeviceInfoSize(Device, OL_DEVICE_INFO_FORCE_UINT32, &Size));
}

TEST_P(olGetDeviceInfoSizeTest, InvalidNullPointer) {
  EXPECT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetDeviceInfoSize(Device, OL_DEVICE_INFO_TYPE, nullptr));
}
