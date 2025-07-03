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
  ASSERT_SUCCESS(olGetDeviceInfoSize(Device, OL_DEVICE_INFO_TYPE, &Size));
  ASSERT_EQ(Size, sizeof(ol_device_type_t));
}

TEST_P(olGetDeviceInfoSizeTest, SuccessPlatform) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(Device, OL_DEVICE_INFO_PLATFORM, &Size));
  ASSERT_EQ(Size, sizeof(ol_platform_handle_t));
}

TEST_P(olGetDeviceInfoSizeTest, SuccessName) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(Device, OL_DEVICE_INFO_NAME, &Size));
  ASSERT_NE(Size, 0ul);
}

TEST_P(olGetDeviceInfoSizeTest, SuccessVendor) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(Device, OL_DEVICE_INFO_VENDOR, &Size));
  ASSERT_NE(Size, 0ul);
}

TEST_P(olGetDeviceInfoSizeTest, SuccessDriverVersion) {
  size_t Size = 0;
  ASSERT_SUCCESS(
      olGetDeviceInfoSize(Device, OL_DEVICE_INFO_DRIVER_VERSION, &Size));
  ASSERT_NE(Size, 0ul);
}

TEST_P(olGetDeviceInfoSizeTest, SuccessMaxWorkGroupSize) {
  size_t Size = 0;
  ASSERT_SUCCESS(
      olGetDeviceInfoSize(Device, OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE, &Size));
  ASSERT_EQ(Size, sizeof(ol_dimensions_t));
  ASSERT_EQ(Size, sizeof(uint32_t) * 3);
}

TEST_P(olGetDeviceInfoSizeTest, InvalidNullHandle) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetDeviceInfoSize(nullptr, OL_DEVICE_INFO_TYPE, &Size));
}

TEST_P(olGetDeviceInfoSizeTest, InvalidDeviceInfoEnumeration) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetDeviceInfoSize(Device, OL_DEVICE_INFO_FORCE_UINT32, &Size));
}

TEST_P(olGetDeviceInfoSizeTest, InvalidNullPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetDeviceInfoSize(Device, OL_DEVICE_INFO_TYPE, nullptr));
}
