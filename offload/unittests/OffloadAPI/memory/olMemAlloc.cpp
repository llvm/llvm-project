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
  ASSERT_SUCCESS(olMemAllocManaged(Device, 1024, &Alloc));
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
  ASSERT_SUCCESS(olMemAllocDevice(Device, 1024, &Alloc));
  ASSERT_NE(Alloc, nullptr);
  olMemFree(Alloc);
}

TEST_P(olMemAllocTest, SuccessAllocMany) {
  std::vector<void *> Allocs;
  Allocs.reserve(1000);

  for (size_t I = 1; I < 1000; I++) {
    void *Alloc = nullptr;
    switch (I % 3) {
    case 0:
      ASSERT_SUCCESS(olMemAllocDevice(Device, 1024 * I, &Alloc));
      break;
    case 1:
      ASSERT_SUCCESS(olMemAllocManaged(Device, 1024 * I, &Alloc));
      break;
    case 2:
      ASSERT_SUCCESS(olMemAllocHost(Device, 1024 * I, &Alloc));
      break;
    }
    ASSERT_NE(Alloc, nullptr);

    Allocs.push_back(Alloc);
  }

  for (auto *A : Allocs) {
    olMemFree(A);
  }
}

TEST_P(olMemAllocTest, InvalidNullDeviceHost) {
  void *Alloc = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olMemAllocHost(nullptr, 1024, &Alloc));
}

TEST_P(olMemAllocTest, InvalidNullDeviceManaged) {
  void *Alloc = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olMemAllocManaged(nullptr, 1024, &Alloc));
}

TEST_P(olMemAllocTest, InvalidNullDeviceDevice) {
  void *Alloc = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olMemAllocDevice(nullptr, 1024, &Alloc));
}

TEST_P(olMemAllocTest, InvalidNullOutPtrHost) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemAllocHost(Device, 1024, nullptr));
}

TEST_P(olMemAllocTest, InvalidNullOutPtrManaged) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemAllocManaged(Device, 1024, nullptr));
}

TEST_P(olMemAllocTest, InvalidNullOutPtrDevice) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemAllocDevice(Device, 1024, nullptr));
}
