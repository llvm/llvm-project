//===------- Offload API tests - olGetDeviceInfoSize -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Properties.hpp"
#include <OffloadAPI.h>

using olGetDeviceInfoSizeTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetDeviceInfoSizeTest);

using olGetDeviceInfoSizeEqualTest = olGetHostDeviceInfoPropertyTest;
using olGetDeviceInfoSizeNonZeroTest = olGetHostDeviceInfoPropertyTest;

DeviceInfoProperties answerSizeEqualToTypeSizeProperties = mergeProperties(
    {Uint32Properties, Uint64Properties, CapabilitesFlagsProperties,
     PlatformProperties, DeviceTypeProperties});

OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE_WITH_PARAM(
    olGetDeviceInfoSizeEqualTest, answerSizeEqualToTypeSizeProperties,
    defaultPropertyTestPrinter<ol_device_info_t>);

OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE_WITH_PARAM(
    olGetDeviceInfoSizeNonZeroTest, NamesProperties,
    defaultPropertyTestPrinter<ol_device_info_t>);

TEST_P(olGetDeviceInfoSizeEqualTest, Success) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(Device, Property, &Size));
  ASSERT_EQ(PropertySize, Size);
}

TEST_P(olGetDeviceInfoSizeNonZeroTest, Success) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(Device, Property, &Size));
  ASSERT_NE(Size, 0ul);
}

TEST_P(olGetDeviceInfoSizeTest, SuccessMaxWorkGroupSizePerDimension) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(
      Device, OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE_PER_DIMENSION, &Size));
  ASSERT_EQ(Size, sizeof(ol_dimensions_t));
  ASSERT_EQ(Size, sizeof(uint32_t) * 3);
}

TEST_P(olGetDeviceInfoSizeTest, SuccessMaxWorkSizePerDimension) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(
      Device, OL_DEVICE_INFO_MAX_WORK_SIZE_PER_DIMENSION, &Size));
  ASSERT_EQ(Size, sizeof(ol_dimensions_t));
  ASSERT_EQ(Size, sizeof(uint32_t) * 3);
}

TEST(olGetDeviceInfoSizeHostTest, SuccessDriverId) {
  ol_device_handle_t Host = TestEnvironment::getHostDevice();
  ASSERT_NE(Host, nullptr);

  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(Host, OL_DEVICE_INFO_DRIVER_ID, &Size));
  ASSERT_EQ(Size, sizeof(uint32_t));
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
