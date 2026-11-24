//===------- Offload API tests - olMemFree --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Properties.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olMemFreeTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemFreeTest);

struct olMemFreeAllocTypesTest : OffloadDeviceTestWithParam<ol_alloc_type_t> {
  ol_result_t allocate(size_t Size, void **Alloc) {
    ol_alloc_type_t AllocType = getTestParam();
    if (AllocType == OL_ALLOC_TYPE_HOST)
      return olMemAllocHost(Context, Device, Size, Alloc);
    return olMemAlloc(Context, Device, AllocType, Size, Alloc);
  }
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE_WITH_PARAM(
    olMemFreeAllocTypesTest, AllocTypes,
    defaultPrinterWithParam<ol_alloc_type_t>);

TEST_P(olMemFreeAllocTypesTest, Success) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(allocate(1024, &Alloc));
  ASSERT_SUCCESS(olMemFree(Context, Alloc));
}

TEST_P(olMemFreeAllocTypesTest, InvalidDoubleFree) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(allocate(1024, &Alloc));
  ASSERT_SUCCESS(olMemFree(Context, Alloc));
  ASSERT_ERROR(OL_ERRC_NOT_FOUND, olMemFree(Context, Alloc));
}

TEST_P(olMemFreeAllocTypesTest, InvalidInteriorPointer) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(allocate(1024, &Alloc));
  ASSERT_ERROR(OL_ERRC_NOT_FOUND,
               olMemFree(Context, static_cast<char *>(Alloc) + 1));
  ASSERT_SUCCESS(olMemFree(Context, Alloc));
}

TEST_P(olMemFreeAllocTypesTest, InvalidOtherContext) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(allocate(1024, &Alloc));

  // Allocations are isolated to the context they were made in, even from a
  // context grouping the same devices.
  ol_context_handle_t OtherContext = nullptr;
  ASSERT_SUCCESS(
      olCreateContext(Devices.size(), Devices.data(), &OtherContext));
  ol_result_t Result = olMemFree(OtherContext, Alloc);
  ASSERT_SUCCESS(olDestroyContext(OtherContext));
  ASSERT_ERROR(OL_ERRC_NOT_FOUND, Result);

  ASSERT_SUCCESS(olMemFree(Context, Alloc));
}

TEST_P(olMemFreeTest, InvalidNotAllocated) {
  int NotAllocated = 0;
  ASSERT_ERROR(OL_ERRC_NOT_FOUND, olMemFree(Context, &NotAllocated));
}

TEST_P(olMemFreeTest, InvalidNullPtr) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, 1024, &Alloc));
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER, olMemFree(Context, nullptr));
  ASSERT_SUCCESS(olMemFree(Context, Alloc));
}
