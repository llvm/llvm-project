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

#define OL_DEVICE_INFO_TEST_SUCCESS_CHECK(TestName, PropType, PropName, Dev,   \
                                          Expr)                                \
  TEST_P(olGetDeviceInfoTest, Test##Dev##TestName) {                           \
    PropType Value;                                                            \
    ASSERT_SUCCESS(olGetDeviceInfo(Dev, PropName, sizeof(Value), &Value));     \
    Expr;                                                                      \
  }

#define OL_DEVICE_INFO_TEST_DEVICE_SUCCESS(TestName, PropType, PropName)       \
  OL_DEVICE_INFO_TEST_SUCCESS_CHECK(TestName, PropType, PropName, Device, {})

#define OL_DEVICE_INFO_TEST_HOST_SUCCESS(TestName, PropType, PropName)         \
  OL_DEVICE_INFO_TEST_SUCCESS_CHECK(TestName, PropType, PropName, Host, {})

#define OL_DEVICE_INFO_TEST_SUCCESS(TestName, PropType, PropName)              \
  OL_DEVICE_INFO_TEST_DEVICE_SUCCESS(TestName, PropType, PropName)             \
  OL_DEVICE_INFO_TEST_HOST_SUCCESS(TestName, PropType, PropName)

#define OL_DEVICE_INFO_TEST_DEVICE_VALUE_GT(TestName, PropType, PropName,      \
                                            LowBound)                          \
  OL_DEVICE_INFO_TEST_SUCCESS_CHECK(TestName, PropType, PropName, Device,      \
                                    ASSERT_GT(Value, LowBound))

#define OL_DEVICE_INFO_TEST_HOST_VALUE_GT(TestName, PropType, PropName,        \
                                          LowBound)                            \
  OL_DEVICE_INFO_TEST_SUCCESS_CHECK(TestName, PropType, PropName, Host,        \
                                    ASSERT_GT(Value, LowBound))

#define OL_DEVICE_INFO_TEST_VALUE_GT(TestName, PropType, PropName, LowBound)   \
  OL_DEVICE_INFO_TEST_DEVICE_VALUE_GT(TestName, PropType, PropName, LowBound)  \
  OL_DEVICE_INFO_TEST_HOST_VALUE_GT(TestName, PropType, PropName, LowBound)

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

TEST_P(olGetDeviceInfoTest, SuccessProductName) {
  size_t Size = 0;
  ASSERT_SUCCESS(
      olGetDeviceInfoSize(Device, OL_DEVICE_INFO_PRODUCT_NAME, &Size));
  ASSERT_GT(Size, 0ul);
  std::vector<char> Name;
  Name.resize(Size);
  ASSERT_SUCCESS(
      olGetDeviceInfo(Device, OL_DEVICE_INFO_PRODUCT_NAME, Size, Name.data()));
  ASSERT_EQ(std::strlen(Name.data()), Size - 1);
}

TEST_P(olGetDeviceInfoTest, HostProductName) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetDeviceInfoSize(Host, OL_DEVICE_INFO_PRODUCT_NAME, &Size));
  ASSERT_GT(Size, 0ul);
  std::vector<char> Name;
  Name.resize(Size);
  ASSERT_SUCCESS(
      olGetDeviceInfo(Host, OL_DEVICE_INFO_PRODUCT_NAME, Size, Name.data()));
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

OL_DEVICE_INFO_TEST_VALUE_GT(MaxWorkGroupSize, uint32_t,
                             OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE, 0);

TEST_P(olGetDeviceInfoTest, SuccessMaxWorkGroupSizePerDimension) {
  ol_dimensions_t Value{0, 0, 0};
  ASSERT_SUCCESS(
      olGetDeviceInfo(Device, OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE_PER_DIMENSION,
                      sizeof(Value), &Value));
  ASSERT_GT(Value.x, 0u);
  ASSERT_GT(Value.y, 0u);
  ASSERT_GT(Value.z, 0u);
}

OL_DEVICE_INFO_TEST_DEVICE_VALUE_GT(VendorId, uint32_t,
                                    OL_DEVICE_INFO_VENDOR_ID, 0);
OL_DEVICE_INFO_TEST_HOST_SUCCESS(VendorId, uint32_t, OL_DEVICE_INFO_VENDOR_ID);
OL_DEVICE_INFO_TEST_VALUE_GT(NumComputeUnits, uint32_t,
                             OL_DEVICE_INFO_NUM_COMPUTE_UNITS, 0);
OL_DEVICE_INFO_TEST_VALUE_GT(SingleFPConfig, ol_device_fp_capability_flags_t,
                             OL_DEVICE_INFO_SINGLE_FP_CONFIG, 0);
OL_DEVICE_INFO_TEST_SUCCESS(HalfFPConfig, ol_device_fp_capability_flags_t,
                            OL_DEVICE_INFO_HALF_FP_CONFIG);
OL_DEVICE_INFO_TEST_VALUE_GT(DoubleFPConfig, ol_device_fp_capability_flags_t,
                             OL_DEVICE_INFO_DOUBLE_FP_CONFIG, 0);
OL_DEVICE_INFO_TEST_VALUE_GT(NativeVectorWidthChar, uint32_t,
                             OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR, 0);
OL_DEVICE_INFO_TEST_VALUE_GT(NativeVectorWidthShort, uint32_t,
                             OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT, 0);
OL_DEVICE_INFO_TEST_VALUE_GT(NativeVectorWidthInt, uint32_t,
                             OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT, 0);
OL_DEVICE_INFO_TEST_VALUE_GT(NativeVectorWidthLong, uint32_t,
                             OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG, 0);
OL_DEVICE_INFO_TEST_VALUE_GT(NativeVectorWidthFloat, uint32_t,
                             OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT, 0);
OL_DEVICE_INFO_TEST_VALUE_GT(NativeVectorWidthDouble, uint32_t,
                             OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE, 0);
OL_DEVICE_INFO_TEST_SUCCESS(NativeVectorWidthHalf, uint32_t,
                            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF);
OL_DEVICE_INFO_TEST_VALUE_GT(MaxClockFrequency, uint32_t,
                             OL_DEVICE_INFO_MAX_CLOCK_FREQUENCY, 0);
OL_DEVICE_INFO_TEST_VALUE_GT(MemoryClockRate, uint32_t,
                             OL_DEVICE_INFO_MEMORY_CLOCK_RATE, 0);
OL_DEVICE_INFO_TEST_VALUE_GT(AddressBits, uint32_t, OL_DEVICE_INFO_ADDRESS_BITS,
                             0);
OL_DEVICE_INFO_TEST_DEVICE_VALUE_GT(MaxMemAllocSize, uint64_t,
                                    OL_DEVICE_INFO_MAX_MEM_ALLOC_SIZE, 0);
OL_DEVICE_INFO_TEST_HOST_SUCCESS(MaxMemAllocSize, uint64_t,
                                 OL_DEVICE_INFO_MAX_MEM_ALLOC_SIZE);
OL_DEVICE_INFO_TEST_DEVICE_VALUE_GT(GlobalMemSize, uint64_t,
                                    OL_DEVICE_INFO_GLOBAL_MEM_SIZE, 0);
OL_DEVICE_INFO_TEST_HOST_SUCCESS(GlobalMemSize, uint64_t,
                                 OL_DEVICE_INFO_GLOBAL_MEM_SIZE);

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
