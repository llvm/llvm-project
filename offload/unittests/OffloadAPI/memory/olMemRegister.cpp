//===------- Offload API tests - olMemRegister ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olMemRegisterTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemRegisterTest);

TEST_P(olMemRegisterTest, SuccessRegister) {
  int Arr[50];
  ol_memory_register_flags_t Flags = {};
  void *PinnedPtr = nullptr;
  ASSERT_SUCCESS(olMemRegister(Device, Arr, sizeof(Arr), Flags, &PinnedPtr));
  ASSERT_NE(PinnedPtr, nullptr);
  ASSERT_SUCCESS(olMemUnregister(Device, PinnedPtr));
}

TEST_P(olMemRegisterTest, SuccessMultipleRegister) {
  int Arr[50];
  ol_memory_register_flags_t Flags = {};
  void *PinnedPtr = nullptr;
  void *PinnedPtr1 = nullptr;
  ASSERT_SUCCESS(olMemRegister(Device, Arr, sizeof(Arr), Flags, &PinnedPtr));
  ASSERT_NE(PinnedPtr, nullptr);
  ASSERT_SUCCESS(olMemRegister(Device, Arr, sizeof(Arr), Flags, &PinnedPtr1));
  ASSERT_NE(PinnedPtr1, nullptr);
  ASSERT_SUCCESS(olMemUnregister(Device, PinnedPtr));
  ASSERT_SUCCESS(olMemUnregister(Device, PinnedPtr1));
}

TEST_P(olMemRegisterTest, InvalidSizeRegister) {
  int Arr[50];
  ol_memory_register_flags_t Flags = {};
  void *PinnedPtr = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olMemRegister(Device, Arr, 0, Flags, &PinnedPtr));
}

TEST_P(olMemRegisterTest, InvalidPtrRegister) {
  int Arr[50];
  ol_memory_register_flags_t Flags = {};
  void *PinnedPtr = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemRegister(Device, nullptr, sizeof(Arr), Flags, &PinnedPtr));
}

TEST_P(olMemRegisterTest, InvalidPtrUnRegister) {
  int Arr[50];
  ol_memory_register_flags_t Flags = {};
  void *PinnedPtr = nullptr;
  ASSERT_SUCCESS(olMemRegister(Device, Arr, sizeof(Arr), Flags, &PinnedPtr));
  ASSERT_NE(PinnedPtr, nullptr);
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER, olMemUnregister(Device, nullptr));
  ASSERT_SUCCESS(olMemUnregister(Device, PinnedPtr));
}

TEST_P(olMemRegisterTest, UnregisteredPtrUnRegister) {
  int Arr[50];
  int Arr1[50];
  ol_memory_register_flags_t Flags = {};
  void *PinnedPtr = nullptr;
  ASSERT_SUCCESS(olMemRegister(Device, Arr, sizeof(Arr), Flags, &PinnedPtr));
  ASSERT_NE(PinnedPtr, nullptr);
  ASSERT_ERROR(OL_ERRC_INVALID_ARGUMENT, olMemUnregister(Device, Arr1));
  ASSERT_SUCCESS(olMemUnregister(Device, PinnedPtr));
}

TEST_P(olMemRegisterTest, PartialOverlapPtrRegister) {
  int Arr[50];
  ol_memory_register_flags_t Flags = {};
  void *PinnedPtr = nullptr;
  ASSERT_SUCCESS(olMemRegister(Device, Arr, sizeof(Arr), Flags, &PinnedPtr));
  ASSERT_NE(PinnedPtr, nullptr);
  ASSERT_ERROR(OL_ERRC_INVALID_ARGUMENT,
               olMemRegister(Device, Arr + 2, sizeof(Arr), Flags, &PinnedPtr));
  ASSERT_SUCCESS(olMemUnregister(Device, PinnedPtr));
}
