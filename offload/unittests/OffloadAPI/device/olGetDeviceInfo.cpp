//===------- Offload API tests - olGetDeviceInfo --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Properties.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

DeviceInfoProperties JustSupportedProperties = mergeProperties(
    {BoolProperties, IrrelevantForHostGTCapabilitiesProperties,
     IrrelevantForHostGTUint32Properties, Uint64MayBeZeroProperties,
     IrrelevantForHostGTUint64Properties});

DeviceInfoProperties NonZeroProperties =
    mergeProperties({RelevantGTCapabilitiesProperties,
                     RelevantGTUint32Properties, RelevantGTUint64Properties});

using olGetHostDeviceInfoPropertySupportTest = olGetHostDeviceInfoPropertyTest;
using olGetHostDeviceInfoPropertyNonZeroTest = olGetHostDeviceInfoPropertyTest;
using olGetDeviceInfoPropertyNonZeroTest = olGetHostDeviceInfoPropertyTest;

OFFLOAD_TESTS_INSTANTIATE_HOST_DEVICE_FIXTURE_WITH_PARAM(
    olGetHostDeviceInfoPropertySupportTest, JustSupportedProperties,
    defaultPropertyTestPrinter<ol_device_info_t>);

OFFLOAD_TESTS_INSTANTIATE_HOST_DEVICE_FIXTURE_WITH_PARAM(
    olGetHostDeviceInfoPropertyNonZeroTest, NonZeroProperties,
    defaultPropertyTestPrinter<ol_device_info_t>);

// Properties that only the host is allowed to report as zero.
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE_WITH_PARAM(
    olGetDeviceInfoPropertyNonZeroTest, IrrelevantForHostGTUint64Properties,
    defaultPropertyTestPrinter<ol_device_info_t>);

// Properties without gt test
TEST_P(olGetHostDeviceInfoPropertySupportTest, Success) {
  char Value[MAX_DEVICE_INFO_BYTES];
  ASSERT_SUCCESS(olGetDeviceInfo(Device, Property, PropertySize, &Value));
}

TEST_P(olGetHostDeviceInfoPropertyNonZeroTest, Value) {
  char Value[MAX_DEVICE_INFO_BYTES] = {};
  ASSERT_SUCCESS(olGetDeviceInfo(Device, Property, PropertySize, &Value));

  ASSERT_TRUE(defaultCheckIsNonZero(Value));
}

TEST_P(olGetDeviceInfoPropertyNonZeroTest, Value) {
  char Value[MAX_DEVICE_INFO_BYTES] = {};
  ASSERT_SUCCESS(olGetDeviceInfo(Device, Property, PropertySize, &Value));

  ASSERT_TRUE(defaultCheckIsNonZero(Value));
}

using olGetDeviceHostInfoNamesTest = olGetHostDeviceInfoPropertyTest;

OFFLOAD_TESTS_INSTANTIATE_HOST_DEVICE_FIXTURE_WITH_PARAM(
    olGetDeviceHostInfoNamesTest, NamesProperties,
    defaultPropertyTestPrinter<ol_device_info_t>);

TEST_P(olGetDeviceHostInfoNamesTest, SuccessNames) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(Device, Property, &Size));
  ASSERT_GT(Size, 0ul);
  std::vector<char> Name;
  Name.resize(Size);
  ASSERT_SUCCESS(olGetDeviceInfo(Device, Property, Size, Name.data()));
  ASSERT_EQ(std::strlen(Name.data()), Size - 1);
}

using olGetHostDeviceInfoDimensionsTest = olGetHostDeviceInfoPropertyTest;

OFFLOAD_TESTS_INSTANTIATE_HOST_DEVICE_FIXTURE_WITH_PARAM(
    olGetHostDeviceInfoDimensionsTest, DimensionsProperties,
    defaultPropertyTestPrinter<ol_device_info_t>);

TEST_P(olGetHostDeviceInfoDimensionsTest, Success) {
  ol_dimensions_t Value{0, 0, 0};
  ASSERT_SUCCESS(olGetDeviceInfo(Device, Property, sizeof(Value), &Value));
  ASSERT_GT(Value.x, 0u);
  ASSERT_GT(Value.y, 0u);
  ASSERT_GT(Value.z, 0u);
}

OFFLOAD_TESTS_INSTANTIATE_HOST_DEVICE_FIXTURE(olGetHostDeviceInfoTest);

TEST_P(olGetHostDeviceInfoTest, HostSuccessType) {
  ol_device_type_t DeviceType;
  ASSERT_SUCCESS(olGetDeviceInfo(Device, OL_DEVICE_INFO_TYPE,
                                 sizeof(ol_device_type_t), &DeviceType));

  if (isHost()) {
    ASSERT_EQ(DeviceType, OL_DEVICE_TYPE_HOST);
  }
}

TEST_P(olGetHostDeviceInfoTest, SuccessPlatform) {
  ol_platform_handle_t Platform = nullptr;
  ASSERT_SUCCESS(olGetDeviceInfo(Device, OL_DEVICE_INFO_PLATFORM,
                                 sizeof(ol_platform_handle_t), &Platform));
  ASSERT_NE(Platform, nullptr);
}

TEST_P(olGetHostDeviceInfoTest, SuccessDriverId) {
  uint32_t DriverId = 0;
  ASSERT_SUCCESS(olGetDeviceInfo(Device, OL_DEVICE_INFO_DRIVER_ID,
                                 sizeof(DriverId), &DriverId));
}

TEST_P(olGetHostDeviceInfoTest, InvalidNullHandleDevice) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetDeviceInfo(nullptr, OL_DEVICE_INFO_TYPE,
                               sizeof(ol_device_type_t), &DeviceType));
}

TEST_P(olGetHostDeviceInfoTest, InvalidEnumerationInfoType) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetDeviceInfo(Device, OL_DEVICE_INFO_FORCE_UINT32,
                               sizeof(ol_device_type_t), &DeviceType));
}

TEST_P(olGetHostDeviceInfoTest, InvalidPropSize) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetDeviceInfo(Device, OL_DEVICE_INFO_TYPE, 0, &DeviceType));
}

TEST_P(olGetHostDeviceInfoTest, InvalidPropSizeSmall) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetDeviceInfo(Device, OL_DEVICE_INFO_TYPE,
                               sizeof(DeviceType) - 1, &DeviceType));
}

TEST_P(olGetHostDeviceInfoTest, InvalidNullPointerPropValue) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetDeviceInfo(Device, OL_DEVICE_INFO_TYPE, sizeof(DeviceType),
                               nullptr));
}

using olGetDeviceInfoNativeHandleTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetDeviceInfoNativeHandleTest);

TEST_P(olGetDeviceInfoNativeHandleTest, SuccessNativeHandle) {
  void *Handle = reinterpret_cast<void *>(~0ull);
  ASSERT_SUCCESS_OR_UNSUPPORTED(olGetDeviceInfo(
      Device, OL_DEVICE_INFO_NATIVE_HANDLE, sizeof(Handle), &Handle));
  // Not all backends report a pointer here; CUDA reports a CUdevice, so the
  // value may legitimately be zero.
  ASSERT_NE(Handle, reinterpret_cast<void *>(~0ull));
}

TEST_P(olGetDeviceInfoNativeHandleTest, SuccessNativePlatformHandle) {
  void *Handle = nullptr;
  ASSERT_SUCCESS_OR_UNSUPPORTED(olGetDeviceInfo(
      Device, OL_DEVICE_INFO_NATIVE_PLATFORM_HANDLE, sizeof(Handle), &Handle));
  ASSERT_NE(Handle, nullptr);
}
