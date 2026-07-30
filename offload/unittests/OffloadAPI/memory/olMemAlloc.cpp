//===------- Offload API tests - olMemAlloc -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olMemAllocTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemAllocTest);

TEST_P(olMemAllocTest, SuccessAllocManaged) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, 1024, &Alloc));
  ASSERT_NE(Alloc, nullptr);
  olMemFree(Alloc);
}

TEST_P(olMemAllocTest, SuccessAllocHost) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAllocHost(Device, 1024, &Alloc));
  ASSERT_NE(Alloc, nullptr);
  olMemFree(Alloc);
}

TEST_P(olMemAllocTest, SuccessAllocDevice) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 1024, &Alloc));
  ASSERT_NE(Alloc, nullptr);
  olMemFree(Alloc);
}

TEST_P(olMemAllocTest, SuccessAllocMany) {
  std::vector<void *> Allocs;
  Allocs.reserve(1000);

  constexpr ol_alloc_type_t TYPES[2] = {OL_ALLOC_TYPE_DEVICE,
                                        OL_ALLOC_TYPE_MANAGED};

  for (size_t I = 1; I < 1000; I++) {
    void *Alloc = nullptr;
    if (I % 3 == 2)
      ASSERT_SUCCESS(olMemAllocHost(Device, 1024 * I, &Alloc));
    else
      ASSERT_SUCCESS(olMemAlloc(Device, TYPES[I % 2], 1024 * I, &Alloc));
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
