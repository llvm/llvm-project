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

template <ol_alloc_type_t Type> struct olMemFreeTestBase : OffloadDeviceTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::SetUp());
    ASSERT_SUCCESS(olMemAlloc(Device, Type, 0x1000, &Alloc));
  }

  void *Alloc;
};

struct olMemFreeDeviceTest : olMemFreeTestBase<OL_ALLOC_TYPE_DEVICE> {};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemFreeDeviceTest);

struct olMemFreeHostTest : olMemFreeTestBase<OL_ALLOC_TYPE_HOST> {};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemFreeHostTest);

struct olMemFreeManagedTest : olMemFreeTestBase<OL_ALLOC_TYPE_MANAGED> {};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemFreeManagedTest);

TEST_P(olMemFreeManagedTest, SuccessFree) {
  ASSERT_SUCCESS(olMemFree(Device, Alloc));
}

TEST_P(olMemFreeManagedTest, SuccessFreeNull) {
  ASSERT_SUCCESS(olMemFree(nullptr, Alloc));
}

TEST_P(olMemFreeHostTest, SuccessFree) {
  ASSERT_SUCCESS(olMemFree(Device, Alloc));
}

TEST_P(olMemFreeHostTest, SuccessFreeNull) {
  ASSERT_SUCCESS(olMemFree(nullptr, Alloc));
}

TEST_P(olMemFreeDeviceTest, SuccessFree) {
  ASSERT_SUCCESS(olMemFree(Device, Alloc));
}

TEST_P(olMemFreeDeviceTest, InvalidNullPtr) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER, olMemFree(Device, nullptr));
}

TEST_P(olMemFreeDeviceTest, InvalidNullDevice) {
  ASSERT_ERROR(OL_ERRC_NOT_FOUND, olMemFree(nullptr, Alloc));
}

TEST_P(olMemFreeDeviceTest, InvalidFreeWrongDevice) {
  ASSERT_ERROR(OL_ERRC_NOT_FOUND,
               olMemFree(TestEnvironment::getHostDevice(), Alloc));
}

TEST_P(olMemFreeHostTest, InvalidFreeWrongDevice) {
  ASSERT_ERROR(OL_ERRC_NOT_FOUND,
               olMemFree(TestEnvironment::getHostDevice(), Alloc));
}

TEST_P(olMemFreeManagedTest, InvalidFreeWrongDevice) {
  ASSERT_ERROR(OL_ERRC_NOT_FOUND,
               olMemFree(TestEnvironment::getHostDevice(), Alloc));
}
