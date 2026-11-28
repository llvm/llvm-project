//===------- Offload API tests - olGetMemInfo -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Properties.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

constexpr size_t SIZE = 1024;

struct olGetMemInfoAllocTypeTest : OffloadDeviceTestWithParam<ol_alloc_type_t> {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(
        OffloadDeviceTestWithParam<ol_alloc_type_t>::SetUp());
    AllocType = getTestParam();
    if (AllocType == OL_ALLOC_TYPE_HOST)
      ASSERT_SUCCESS(olMemAllocHost(Context, Device, SIZE, &Ptr));
    else
      ASSERT_SUCCESS(olMemAlloc(Context, Device, AllocType, SIZE, &Ptr));
  }

  void TearDown() override {
    ASSERT_SUCCESS(olMemFree(Context, Ptr));
    RETURN_ON_FATAL_FAILURE(
        OffloadDeviceTestWithParam<ol_alloc_type_t>::TearDown());
  }

  void *Ptr;

  ol_alloc_type_t AllocType;
};

struct olGetMemInfoTest : OffloadDeviceTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::SetUp());
    ASSERT_SUCCESS(
        olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, SIZE, &Ptr));
  }

  void TearDown() override {
    ASSERT_SUCCESS(olMemFree(Context, Ptr));
    RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::TearDown());
  }

  void *Ptr;
};

using olGetMemInfoMultiDeviceTest = OffloadDeviceTest;

OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE_WITH_PARAM(
    olGetMemInfoAllocTypeTest, AllocTypes,
    defaultPrinterWithParam<ol_alloc_type_t>);
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetMemInfoTest);
OFFLOAD_TESTS_INSTANTIATE_MULTI_DEVICE_FIXTURE(olGetMemInfoMultiDeviceTest);

TEST_P(olGetMemInfoAllocTypeTest, SuccessDevice) {
  // Host-pool allocations have no per-device affinity, so querying
  // OL_MEM_INFO_DEVICE is invalid for them.
  ol_device_handle_t RetrievedDevice;
  if (AllocType == OL_ALLOC_TYPE_HOST) {
    ASSERT_ERROR(OL_ERRC_INVALID_ARGUMENT,
                 olGetMemInfo(Context, Ptr, OL_MEM_INFO_DEVICE,
                              sizeof(RetrievedDevice), &RetrievedDevice));
    return;
  }
  ASSERT_SUCCESS(olGetMemInfo(Context, Ptr, OL_MEM_INFO_DEVICE,
                              sizeof(RetrievedDevice), &RetrievedDevice));
  ASSERT_EQ(RetrievedDevice, Device);
}

TEST_P(olGetMemInfoAllocTypeTest, SuccessBase) {
  void *RetrievedBase;
  ASSERT_SUCCESS(olGetMemInfo(Context, Ptr, OL_MEM_INFO_BASE,
                              sizeof(RetrievedBase), &RetrievedBase));
  ASSERT_EQ(RetrievedBase, Ptr);
}

TEST_P(olGetMemInfoAllocTypeTest, SuccessSize) {
  size_t RetrievedSize;
  ASSERT_SUCCESS(olGetMemInfo(Context, Ptr, OL_MEM_INFO_SIZE,
                              sizeof(RetrievedSize), &RetrievedSize));
  ASSERT_EQ(RetrievedSize, SIZE);
}

TEST_P(olGetMemInfoAllocTypeTest, SuccessType) {
  ol_alloc_type_t RetrievedType;
  ASSERT_SUCCESS(olGetMemInfo(Context, Ptr, OL_MEM_INFO_TYPE,
                              sizeof(RetrievedType), &RetrievedType));
  ASSERT_EQ(RetrievedType, getTestParam());
}

TEST_P(olGetMemInfoAllocTypeTest, SuccessInteriorPointer) {
  // The spec allows Ptr to point anywhere inside the allocation; the query
  // must still resolve to the allocation's base and full size.
  void *Interior = reinterpret_cast<char *>(Ptr) + SIZE / 2;

  void *RetrievedBase;
  ASSERT_SUCCESS(olGetMemInfo(Context, Interior, OL_MEM_INFO_BASE,
                              sizeof(RetrievedBase), &RetrievedBase));
  ASSERT_EQ(RetrievedBase, Ptr);

  size_t RetrievedSize;
  ASSERT_SUCCESS(olGetMemInfo(Context, Interior, OL_MEM_INFO_SIZE,
                              sizeof(RetrievedSize), &RetrievedSize));
  ASSERT_EQ(RetrievedSize, SIZE);
}

TEST_P(olGetMemInfoAllocTypeTest, InvalidPastTheEnd) {
  void *RetrievedBase;
  ASSERT_ERROR(OL_ERRC_NOT_FOUND,
               olGetMemInfo(Context, reinterpret_cast<char *>(Ptr) + SIZE,
                            OL_MEM_INFO_BASE, sizeof(RetrievedBase),
                            &RetrievedBase));
}

TEST_P(olGetMemInfoAllocTypeTest, InvalidOtherContext) {
  // Allocations are isolated to the context they were made in, even from a
  // context grouping the same devices.
  ol_context_handle_t OtherContext = nullptr;
  ASSERT_SUCCESS(
      olCreateContext(Devices.size(), Devices.data(), &OtherContext));
  void *RetrievedBase;
  ol_result_t Result = olGetMemInfo(OtherContext, Ptr, OL_MEM_INFO_BASE,
                                    sizeof(RetrievedBase), &RetrievedBase);
  ASSERT_SUCCESS(olDestroyContext(OtherContext));
  ASSERT_ERROR(OL_ERRC_NOT_FOUND, Result);
}

TEST_P(olGetMemInfoMultiDeviceTest, SuccessDeviceOfEachDevice) {
  for (ol_device_handle_t AllocDevice : Devices) {
    void *Ptr = nullptr;
    ASSERT_SUCCESS(
        olMemAlloc(Context, AllocDevice, OL_ALLOC_TYPE_DEVICE, SIZE, &Ptr));
    ol_device_handle_t RetrievedDevice = nullptr;
    ol_result_t Result =
        olGetMemInfo(Context, Ptr, OL_MEM_INFO_DEVICE, sizeof(RetrievedDevice),
                     &RetrievedDevice);
    ASSERT_SUCCESS(olMemFree(Context, Ptr));
    ASSERT_SUCCESS(Result);
    ASSERT_EQ(RetrievedDevice, AllocDevice);
  }
}

TEST_P(olGetMemInfoTest, InvalidNotFound) {
  // Assuming that we aren't unlucky and happen to get 0x1234 as a random
  // pointer
  void *RetrievedBase;
  ASSERT_ERROR(OL_ERRC_NOT_FOUND,
               olGetMemInfo(Context, reinterpret_cast<void *>(0x1234),
                            OL_MEM_INFO_BASE, sizeof(RetrievedBase),
                            &RetrievedBase));
}

TEST_P(olGetMemInfoTest, InvalidNullPtr) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetMemInfo(Context, nullptr, OL_MEM_INFO_DEVICE,
                            sizeof(RetrievedDevice), &RetrievedDevice));
}

TEST_P(olGetMemInfoTest, InvalidSizeZero) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_ERROR(
      OL_ERRC_INVALID_SIZE,
      olGetMemInfo(Context, Ptr, OL_MEM_INFO_DEVICE, 0, &RetrievedDevice));
}

TEST_P(olGetMemInfoTest, InvalidSizeSmall) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetMemInfo(Context, Ptr, OL_MEM_INFO_DEVICE,
                            sizeof(RetrievedDevice) - 1, &RetrievedDevice));
}

TEST_P(olGetMemInfoTest, InvalidNullPointerPropValue) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetMemInfo(Context, Ptr, OL_MEM_INFO_DEVICE,
                            sizeof(RetrievedDevice), nullptr));
}
