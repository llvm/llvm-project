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

#define OL_DEVICE_INFO_SIZE_TEST(TestName, PropName, Expr)                     \
  TEST_P(olGetDeviceInfoSizeTest, Success##TestName) {                         \
    size_t Size = 0;                                                           \
    ASSERT_SUCCESS(olGetDeviceInfoSize(Device, PropName, &Size));              \
    Expr;                                                                      \
  }

#define OL_DEVICE_INFO_SIZE_TEST_EQ(TestName, PropType, PropName)              \
  OL_DEVICE_INFO_SIZE_TEST(TestName, PropName,                                 \
                           ASSERT_EQ(Size, sizeof(PropType)));

#define OL_DEVICE_INFO_SIZE_TEST_NONZERO(TestName, PropName)                   \
  OL_DEVICE_INFO_SIZE_TEST(TestName, PropName, ASSERT_NE(Size, 0ul));

OL_DEVICE_INFO_SIZE_TEST_EQ(Type, ol_device_type_t, OL_DEVICE_INFO_TYPE);
OL_DEVICE_INFO_SIZE_TEST_EQ(Platform, ol_platform_handle_t,
                            OL_DEVICE_INFO_PLATFORM);
OL_DEVICE_INFO_SIZE_TEST_NONZERO(Name, OL_DEVICE_INFO_NAME);
OL_DEVICE_INFO_SIZE_TEST_NONZERO(ProductName, OL_DEVICE_INFO_PRODUCT_NAME);
OL_DEVICE_INFO_SIZE_TEST_NONZERO(Vendor, OL_DEVICE_INFO_VENDOR);
OL_DEVICE_INFO_SIZE_TEST_NONZERO(DriverVersion, OL_DEVICE_INFO_DRIVER_VERSION);
OL_DEVICE_INFO_SIZE_TEST_EQ(MaxWorkGroupSize, uint32_t,
                            OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE);
OL_DEVICE_INFO_SIZE_TEST_EQ(MaxWorkSize, uint32_t,
                            OL_DEVICE_INFO_MAX_WORK_SIZE);
OL_DEVICE_INFO_SIZE_TEST_EQ(VendorId, uint32_t, OL_DEVICE_INFO_VENDOR_ID);
OL_DEVICE_INFO_SIZE_TEST_EQ(NumComputeUnits, uint32_t,
                            OL_DEVICE_INFO_NUM_COMPUTE_UNITS);
OL_DEVICE_INFO_SIZE_TEST_EQ(SingleFPConfig, ol_device_fp_capability_flags_t,
                            OL_DEVICE_INFO_SINGLE_FP_CONFIG);
OL_DEVICE_INFO_SIZE_TEST_EQ(HalfFPConfig, ol_device_fp_capability_flags_t,
                            OL_DEVICE_INFO_HALF_FP_CONFIG);
OL_DEVICE_INFO_SIZE_TEST_EQ(DoubleFPConfig, ol_device_fp_capability_flags_t,
                            OL_DEVICE_INFO_DOUBLE_FP_CONFIG);
OL_DEVICE_INFO_SIZE_TEST_EQ(NativeVectorWidthChar, uint32_t,
                            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR);
OL_DEVICE_INFO_SIZE_TEST_EQ(NativeVectorWidthShort, uint32_t,
                            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT);
OL_DEVICE_INFO_SIZE_TEST_EQ(NativeVectorWidthInt, uint32_t,
                            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT);
OL_DEVICE_INFO_SIZE_TEST_EQ(NativeVectorWidthLong, uint32_t,
                            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG);
OL_DEVICE_INFO_SIZE_TEST_EQ(NativeVectorWidthFloat, uint32_t,
                            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT);
OL_DEVICE_INFO_SIZE_TEST_EQ(NativeVectorWidthDouble, uint32_t,
                            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE);
OL_DEVICE_INFO_SIZE_TEST_EQ(NativeVectorWidthHalf, uint32_t,
                            OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF);
OL_DEVICE_INFO_SIZE_TEST_EQ(MaxClockFrequency, uint32_t,
                            OL_DEVICE_INFO_MAX_CLOCK_FREQUENCY);
OL_DEVICE_INFO_SIZE_TEST_EQ(MemoryClockRate, uint32_t,
                            OL_DEVICE_INFO_MEMORY_CLOCK_RATE);
OL_DEVICE_INFO_SIZE_TEST_EQ(AddressBits, uint32_t, OL_DEVICE_INFO_ADDRESS_BITS);
OL_DEVICE_INFO_SIZE_TEST_EQ(MaxMemAllocSize, uint64_t,
                            OL_DEVICE_INFO_MAX_MEM_ALLOC_SIZE);
OL_DEVICE_INFO_SIZE_TEST_EQ(GlobalMemSize, uint64_t,
                            OL_DEVICE_INFO_GLOBAL_MEM_SIZE);

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
