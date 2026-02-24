//===------- Offload API tests - olGetHostInfo ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olGetHostInfoTest = OffloadTest;

#define OL_DEVICE_INFO_TEST_SUCCESS_CHECK(TestName, PropType, PropName, Dev,   \
                                          Expr)                                \
  TEST_F(olGetHostInfoTest, Test##Dev##TestName) {                             \
    PropType Value;                                                            \
    ASSERT_SUCCESS(olGetDeviceInfo(Dev, PropName, sizeof(Value), &Value));     \
    Expr;                                                                      \
  }

#define OL_DEVICE_INFO_TEST_HOST_SUCCESS(TestName, PropType, PropName)         \
  OL_DEVICE_INFO_TEST_SUCCESS_CHECK(TestName, PropType, PropName, Host, {})

#define OL_DEVICE_INFO_TEST_HOST_VALUE_GT(TestName, PropType, PropName,        \
                                          LowBound)                            \
  OL_DEVICE_INFO_TEST_SUCCESS_CHECK(TestName, PropType, PropName, Host,        \
                                    ASSERT_GT(Value, LowBound))

TEST_F(olGetHostInfoTest, HostSuccessType) {
  ol_device_type_t DeviceType;
  ASSERT_SUCCESS(olGetDeviceInfo(Host, OL_DEVICE_INFO_TYPE,
                                 sizeof(ol_device_type_t), &DeviceType));
  ASSERT_EQ(DeviceType, OL_DEVICE_TYPE_HOST);
}

TEST_F(olGetHostInfoTest, SuccessHostPlatform) {
  ol_platform_handle_t Platform = nullptr;
  ASSERT_SUCCESS(olGetDeviceInfo(Host, OL_DEVICE_INFO_PLATFORM,
                                 sizeof(ol_platform_handle_t), &Platform));
  ASSERT_NE(Platform, nullptr);
}

TEST_F(olGetHostInfoTest, HostName) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(Host, OL_DEVICE_INFO_NAME, &Size));
  ASSERT_GT(Size, 0ul);
  std::vector<char> Name;
  Name.resize(Size);
  ASSERT_SUCCESS(olGetDeviceInfo(Host, OL_DEVICE_INFO_NAME, Size, Name.data()));
  ASSERT_EQ(std::strlen(Name.data()), Size - 1);
}

TEST_F(olGetHostInfoTest, HostProductName) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(Host, OL_DEVICE_INFO_PRODUCT_NAME, &Size));
  ASSERT_GT(Size, 0ul);
  std::vector<char> Name;
  Name.resize(Size);
  ASSERT_SUCCESS(
      olGetDeviceInfo(Host, OL_DEVICE_INFO_PRODUCT_NAME, Size, Name.data()));
  ASSERT_EQ(std::strlen(Name.data()), Size - 1);
}

TEST_F(olGetHostInfoTest, HostUID) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(Host, OL_DEVICE_INFO_UID, &Size));
  ASSERT_GT(Size, 0ul);
  std::vector<char> UID;
  UID.resize(Size);
  ASSERT_SUCCESS(olGetDeviceInfo(Host, OL_DEVICE_INFO_UID, Size, UID.data()));
  ASSERT_EQ(std::strlen(UID.data()), Size - 1);
}

TEST_F(olGetHostInfoTest, SuccessHostVendor) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(Host, OL_DEVICE_INFO_VENDOR, &Size));
  ASSERT_GT(Size, 0ul);
  std::vector<char> Vendor;
  Vendor.resize(Size);
  ASSERT_SUCCESS(
      olGetDeviceInfo(Host, OL_DEVICE_INFO_VENDOR, Size, Vendor.data()));
  ASSERT_EQ(std::strlen(Vendor.data()), Size - 1);
}

TEST_F(olGetHostInfoTest, SuccessHostDriverVersion) {
  size_t Size = 0;
  ASSERT_SUCCESS(
      olGetDeviceInfoSize(Host, OL_DEVICE_INFO_DRIVER_VERSION, &Size));
  ASSERT_GT(Size, 0ul);
  std::vector<char> DriverVersion;
  DriverVersion.resize(Size);
  ASSERT_SUCCESS(olGetDeviceInfo(Host, OL_DEVICE_INFO_DRIVER_VERSION, Size,
                                 DriverVersion.data()));
  ASSERT_EQ(std::strlen(DriverVersion.data()), Size - 1);
}

OL_DEVICE_INFO_TEST_HOST_VALUE_GT(MaxWorkGroupSize, uint32_t,
                                  OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE, 0);

TEST_F(olGetHostInfoTest, SuccessHostMaxWorkGroupSizePerDimension) {
  ol_dimensions_t Value{0, 0, 0};
  ASSERT_SUCCESS(
      olGetDeviceInfo(Host, OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE_PER_DIMENSION,
                      sizeof(Value), &Value));
  ASSERT_GT(Value.x, 0u);
  ASSERT_GT(Value.y, 0u);
  ASSERT_GT(Value.z, 0u);
}

OL_DEVICE_INFO_TEST_HOST_VALUE_GT(MaxWorkSize, uint32_t,
                                  OL_DEVICE_INFO_MAX_WORK_SIZE, 0);

TEST_F(olGetHostInfoTest, SuccessHostMaxWorkSizePerDimension) {
  ol_dimensions_t Value{0, 0, 0};
  ASSERT_SUCCESS(olGetDeviceInfo(
      Host, OL_DEVICE_INFO_MAX_WORK_SIZE_PER_DIMENSION, sizeof(Value), &Value));
  ASSERT_GT(Value.x, 0u);
  ASSERT_GT(Value.y, 0u);
  ASSERT_GT(Value.z, 0u);
}

OL_DEVICE_INFO_TEST_HOST_VALUE_GT(VendorId, uint32_t, OL_DEVICE_INFO_VENDOR_ID,
                                  0);
OL_DEVICE_INFO_TEST_HOST_VALUE_GT(NumComputeUnits, uint32_t,
                                  OL_DEVICE_INFO_NUM_COMPUTE_UNITS, 0);
OL_DEVICE_INFO_TEST_HOST_VALUE_GT(SingleFPConfig,
                                  ol_device_fp_capability_flags_t,
                                  OL_DEVICE_INFO_SINGLE_FP_CONFIG, 0);
OL_DEVICE_INFO_TEST_HOST_SUCCESS(HalfFPConfig, ol_device_fp_capability_flags_t,
                                 OL_DEVICE_INFO_HALF_FP_CONFIG);
OL_DEVICE_INFO_TEST_HOST_VALUE_GT(DoubleFPConfig,
                                  ol_device_fp_capability_flags_t,
                                  OL_DEVICE_INFO_DOUBLE_FP_CONFIG, 0);
OL_DEVICE_INFO_TEST_HOST_VALUE_GT(NativeVectorWidthChar, uint32_t,
                                  OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR, 0);
OL_DEVICE_INFO_TEST_HOST_VALUE_GT(NativeVectorWidthShort, uint32_t,
                                  OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT, 0);
OL_DEVICE_INFO_TEST_HOST_VALUE_GT(NativeVectorWidthInt, uint32_t,
                                  OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT, 0);
OL_DEVICE_INFO_TEST_HOST_VALUE_GT(NativeVectorWidthLong, uint32_t,
                                  OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG, 0);
OL_DEVICE_INFO_TEST_HOST_VALUE_GT(NativeVectorWidthFloat, uint32_t,
                                  OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT, 0);
OL_DEVICE_INFO_TEST_HOST_VALUE_GT(NativeVectorWidthDouble, uint32_t,
                                  OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE, 0);
OL_DEVICE_INFO_TEST_HOST_SUCCESS(NativeVectorWidthHalf, uint32_t,
                                 OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF);
OL_DEVICE_INFO_TEST_HOST_VALUE_GT(MaxClockFrequency, uint32_t,
                                  OL_DEVICE_INFO_MAX_CLOCK_FREQUENCY, 0);
OL_DEVICE_INFO_TEST_HOST_VALUE_GT(MemoryClockRate, uint32_t,
                                  OL_DEVICE_INFO_MEMORY_CLOCK_RATE, 0);
OL_DEVICE_INFO_TEST_HOST_VALUE_GT(AddressBits, uint32_t,
                                  OL_DEVICE_INFO_ADDRESS_BITS, 0);
OL_DEVICE_INFO_TEST_HOST_VALUE_GT(MaxMemAllocSize, uint64_t,
                                  OL_DEVICE_INFO_MAX_MEM_ALLOC_SIZE, 0);
OL_DEVICE_INFO_TEST_HOST_VALUE_GT(GlobalMemSize, uint64_t,
                                  OL_DEVICE_INFO_GLOBAL_MEM_SIZE, 0);
OL_DEVICE_INFO_TEST_HOST_VALUE_GT(SharedMemSize, uint64_t,
                                  OL_DEVICE_INFO_WORK_GROUP_LOCAL_MEM_SIZE, 0);

TEST_F(olGetHostInfoTest, InvalidNullHandleDevice) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetDeviceInfo(nullptr, OL_DEVICE_INFO_TYPE,
                               sizeof(ol_device_type_t), &DeviceType));
}

TEST_F(olGetHostInfoTest, InvalidEnumerationInfoType) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetDeviceInfo(Host, OL_DEVICE_INFO_FORCE_UINT32,
                               sizeof(ol_device_type_t), &DeviceType));
}

TEST_F(olGetHostInfoTest, InvalidSizePropSize) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetDeviceInfo(Host, OL_DEVICE_INFO_TYPE, 0, &DeviceType));
}

TEST_F(olGetHostInfoTest, InvalidSizePropSizeSmall) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetDeviceInfo(Host, OL_DEVICE_INFO_TYPE,
                               sizeof(DeviceType) - 1, &DeviceType));
}

TEST_F(olGetHostInfoTest, InvalidNullPointerPropValue) {
  ol_device_type_t DeviceType;
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_POINTER,
      olGetDeviceInfo(Host, OL_DEVICE_INFO_TYPE, sizeof(DeviceType), nullptr));
}
