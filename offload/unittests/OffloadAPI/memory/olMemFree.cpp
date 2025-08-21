//===------- Offload API tests - olMemFree --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olMemFreeTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemFreeTest);

TEST_P(olMemFreeTest, SuccessFreeManaged) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, 1024, &Alloc));
  ASSERT_SUCCESS(olMemFree(Device, Alloc));
}

TEST_P(olMemFreeTest, SuccessFreeManagedNull) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, 1024, &Alloc));
  ASSERT_SUCCESS(olMemFree(nullptr, Alloc));
}

TEST_P(olMemFreeTest, SuccessFreeHost) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_HOST, 1024, &Alloc));
  ASSERT_SUCCESS(olMemFree(Device, Alloc));
}

TEST_P(olMemFreeTest, SuccessFreeHostNull) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_HOST, 1024, &Alloc));
  ASSERT_SUCCESS(olMemFree(nullptr, Alloc));
}

TEST_P(olMemFreeTest, SuccessFreeDevice) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 1024, &Alloc));
  ASSERT_SUCCESS(olMemFree(Device, Alloc));
}

TEST_P(olMemFreeTest, InvalidNullPtr) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER, olMemFree(Device, nullptr));
}

TEST_P(olMemFreeTest, InvalidFreeDeviceNull) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 1024, &Alloc));
  ASSERT_ERROR(OL_ERRC_NOT_FOUND, olMemFree(nullptr, Alloc));
}

TEST_P(olMemFreeTest, InvalidFreeManagedWrongDevice) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, 1024, &Alloc));
  ASSERT_ERROR(OL_ERRC_NOT_FOUND,
               olMemFree(TestEnvironment::getHostDevice(), Alloc));
}

TEST_P(olMemFreeTest, InvalidFreeHostWrongDevice) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_HOST, 1024, &Alloc));
  ASSERT_ERROR(OL_ERRC_NOT_FOUND,
               olMemFree(TestEnvironment::getHostDevice(), Alloc));
}

TEST_P(olMemFreeTest, InvalidFreeDeviceWrongDevice) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 1024, &Alloc));
  ASSERT_ERROR(OL_ERRC_NOT_FOUND,
               olMemFree(TestEnvironment::getHostDevice(), Alloc));
}
