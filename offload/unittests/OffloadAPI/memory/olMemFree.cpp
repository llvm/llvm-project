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

using olMemFreeAllocTypesTest = OffloadDeviceTestWithParam<ol_alloc_type_t>;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE_WITH_PARAM(
    olMemFreeAllocTypesTest, AllocTypes,
    defaultPrinterWithParam<ol_alloc_type_t>);

TEST_P(olMemFreeAllocTypesTest, Success) {
  void *Alloc = nullptr;
  ol_alloc_type_t AllocType = getTestParam();
  if (AllocType == OL_ALLOC_TYPE_HOST) {
    ASSERT_SUCCESS(olMemAllocHost(Device, 1024, &Alloc));
  } else {
    ASSERT_SUCCESS(olMemAlloc(Device, AllocType, 1024, &Alloc));
  }
  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_P(olMemFreeTest, InvalidNullPtr) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 1024, &Alloc));
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER, olMemFree(nullptr));
  ASSERT_SUCCESS(olMemFree(Alloc));
}
