//===------- Offload API tests - olMemAlloc -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Properties.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olMemAllocTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemAllocTest);

struct olMemAllocAllocTypesTest : OffloadDeviceTestWithParam<ol_alloc_type_t> {
  ol_result_t allocateDeviceOrHost(size_t Size, void **Alloc) {
    ol_alloc_type_t AllocType = getTestParam();
    if (AllocType == OL_ALLOC_TYPE_HOST) {
      return olMemAllocHost(this->Device, Size, Alloc);
    }

    return olMemAlloc(this->Device, AllocType, Size, Alloc);
  }
};

OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE_WITH_PARAM(
    olMemAllocAllocTypesTest, AllocTypes,
    defaultPrinterWithParam<ol_alloc_type_t>); // printerMine);

TEST_P(olMemAllocAllocTypesTest, Success) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(allocateDeviceOrHost(DefaultAllocSize, &Alloc));
  ASSERT_NE(Alloc, nullptr);
  olMemFree(Alloc);
}

TEST_P(olMemAllocTest, SuccessAllocMany) {
  std::vector<void *> Allocs;
  Allocs.reserve(TestAllocsNum);

  for (size_t I = 1; I < TestAllocsNum; I++) {
    void *Alloc = nullptr;
    ol_alloc_type_t AllocType = AllocTypes[I % 3];
    if (AllocType == OL_ALLOC_TYPE_HOST) {
      ASSERT_SUCCESS(olMemAllocHost(Device, DefaultAllocSize * I, &Alloc));
    } else {
      ASSERT_SUCCESS(
          olMemAlloc(Device, AllocType, DefaultAllocSize * I, &Alloc));
    }
    ASSERT_NE(Alloc, nullptr);

    Allocs.push_back(Alloc);
  }

  for (auto *A : Allocs) {
    olMemFree(A);
  }
}

TEST_P(olMemAllocTest, InvalidNullDevice) {
  void *Alloc = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olMemAlloc(nullptr, OL_ALLOC_TYPE_DEVICE, 1024, &Alloc));
}

TEST_P(olMemAllocTest, InvalidNullDeviceHost) {
  void *Alloc = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olMemAllocHost(nullptr, 1024, &Alloc));
}

TEST_P(olMemAllocTest, InvalidNullOutPtr) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 1024, nullptr));
}

TEST_P(olMemAllocTest, InvalidNullOutPtrHost) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemAllocHost(Device, 1024, nullptr));
}

TEST_P(olMemAllocTest, InvalidHostType) {
  void *Alloc = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olMemAlloc(Device, OL_ALLOC_TYPE_HOST, 1024, &Alloc));
}
