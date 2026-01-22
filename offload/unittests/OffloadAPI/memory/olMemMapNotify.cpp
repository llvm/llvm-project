//===------- Offload API tests - olMemMapNotify -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olMemMapNotifyTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemMapNotifyTest);

TEST_P(olMemMapNotifyTest, SuccessMapNotify) {
  int Arr[50];

  ASSERT_SUCCESS(olMemDataMappedNotify(Device, Arr, sizeof(Arr)));
  ASSERT_SUCCESS(olMemDataUnMappedNotify(Device, Arr));
}

TEST_P(olMemMapNotifyTest, SuccessMultipleMapNotify) {
  int Arr[50];
  ASSERT_SUCCESS(olMemDataMappedNotify(Device, Arr, sizeof(Arr)));
  ASSERT_SUCCESS(olMemDataMappedNotify(Device, Arr, sizeof(Arr)));
  ASSERT_SUCCESS(olMemDataUnMappedNotify(Device, Arr));
  ASSERT_SUCCESS(olMemDataUnMappedNotify(Device, Arr));
}

TEST_P(olMemMapNotifyTest, InvalidSizeMapNotify) {
  int Arr[50];
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE, olMemDataMappedNotify(Device, Arr, 0));
}

TEST_P(olMemMapNotifyTest, InvalidPtrMapNotify) {
  int Arr[50];
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemDataMappedNotify(Device, nullptr, sizeof(Arr)));
}

TEST_P(olMemMapNotifyTest, InvalidPtrUnMapNotify) {
  int Arr[50];
  ASSERT_SUCCESS(olMemDataMappedNotify(Device, Arr, sizeof(Arr)));
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemDataUnMappedNotify(Device, nullptr));
  ASSERT_SUCCESS(olMemDataUnMappedNotify(Device, Arr));
}
