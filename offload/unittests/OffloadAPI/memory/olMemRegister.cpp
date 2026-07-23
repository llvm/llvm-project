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
  ol_memory_register_flags_t FlagsReg = OL_MEMORY_REGISTER_FLAG_LOCK_MEMORY;
  ol_memory_register_flags_t FlagsUnreg = OL_MEMORY_REGISTER_FLAG_UNLOCK_MEMORY;
  void *PinnedPtr = nullptr;
  ASSERT_SUCCESS(olMemRegister(Device, Arr, sizeof(Arr), FlagsReg, &PinnedPtr));
  ASSERT_NE(PinnedPtr, nullptr);
  ASSERT_SUCCESS(olMemUnregister(Device, PinnedPtr, FlagsUnreg));
}

TEST_P(olMemRegisterTest, SuccessMultipleRegister) {
  int Arr[50];
  ol_memory_register_flags_t FlagsReg = OL_MEMORY_REGISTER_FLAG_LOCK_MEMORY;
  ol_memory_register_flags_t FlagsUnreg = OL_MEMORY_REGISTER_FLAG_UNLOCK_MEMORY;
  void *PinnedPtr = nullptr;
  void *PinnedPtr1 = nullptr;
  ASSERT_SUCCESS(olMemRegister(Device, Arr, sizeof(Arr), FlagsReg, &PinnedPtr));
  ASSERT_NE(PinnedPtr, nullptr);
  ASSERT_SUCCESS(
      olMemRegister(Device, Arr, sizeof(Arr), FlagsReg, &PinnedPtr1));
  ASSERT_NE(PinnedPtr1, nullptr);
  ASSERT_SUCCESS(olMemUnregister(Device, PinnedPtr, FlagsUnreg));
  ASSERT_SUCCESS(olMemUnregister(Device, PinnedPtr1, FlagsUnreg));
}

TEST_P(olMemRegisterTest, InvalidSizeRegister) {
  int Arr[50];
  ol_memory_register_flags_t Flags = OL_MEMORY_REGISTER_FLAG_LOCK_MEMORY;
  void *PinnedPtr = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olMemRegister(Device, Arr, 0, Flags, &PinnedPtr));
}

TEST_P(olMemRegisterTest, InvalidPtrRegister) {
  int Arr[50];
  ol_memory_register_flags_t Flags = OL_MEMORY_REGISTER_FLAG_LOCK_MEMORY;
  void *PinnedPtr = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemRegister(Device, nullptr, sizeof(Arr), Flags, &PinnedPtr));
}

TEST_P(olMemRegisterTest, InvalidPtrUnRegister) {
  int Arr[50];
  ol_memory_register_flags_t FlagsReg = OL_MEMORY_REGISTER_FLAG_LOCK_MEMORY;
  ol_memory_register_flags_t FlagsUnreg = OL_MEMORY_REGISTER_FLAG_UNLOCK_MEMORY;
  void *PinnedPtr = nullptr;
  ASSERT_SUCCESS(olMemRegister(Device, Arr, sizeof(Arr), FlagsReg, &PinnedPtr));
  ASSERT_NE(PinnedPtr, nullptr);
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemUnregister(Device, nullptr, FlagsUnreg));
  ASSERT_SUCCESS(olMemUnregister(Device, PinnedPtr, FlagsUnreg));
}

TEST_P(olMemRegisterTest, UnregisteredPtrUnRegister) {
  int Arr[50];
  int Arr1[50];
  ol_memory_register_flags_t FlagsReg = OL_MEMORY_REGISTER_FLAG_LOCK_MEMORY;
  ol_memory_register_flags_t FlagsUnreg = OL_MEMORY_REGISTER_FLAG_UNLOCK_MEMORY;
  void *PinnedPtr = nullptr;
  ASSERT_SUCCESS(olMemRegister(Device, Arr, sizeof(Arr), FlagsReg, &PinnedPtr));
  ASSERT_NE(PinnedPtr, nullptr);
  ASSERT_ERROR(OL_ERRC_INVALID_ARGUMENT,
               olMemUnregister(Device, Arr1, FlagsUnreg));
  ASSERT_SUCCESS(olMemUnregister(Device, PinnedPtr, FlagsUnreg));
}

TEST_P(olMemRegisterTest, PartialOverlapPtrRegister) {
  int Arr[50];
  ol_memory_register_flags_t FlagsReg = OL_MEMORY_REGISTER_FLAG_LOCK_MEMORY;
  ol_memory_register_flags_t FlagsUnreg = OL_MEMORY_REGISTER_FLAG_UNLOCK_MEMORY;
  void *PinnedPtr = nullptr;
  ASSERT_SUCCESS(olMemRegister(Device, Arr, sizeof(Arr), FlagsReg, &PinnedPtr));
  ASSERT_NE(PinnedPtr, nullptr);
  ASSERT_ERROR(
      OL_ERRC_INVALID_ARGUMENT,
      olMemRegister(Device, Arr + 2, sizeof(Arr), FlagsReg, &PinnedPtr));
  ASSERT_SUCCESS(olMemUnregister(Device, PinnedPtr, FlagsUnreg));
}

TEST_P(olMemRegisterTest, SuccessRegisterNoLock) {
  int Arr[50];
  ol_memory_register_flags_t Flags = {0};
  void *PinnedPtr = nullptr;
  ASSERT_SUCCESS(olMemRegister(Device, Arr, sizeof(Arr), Flags, &PinnedPtr));
  ASSERT_SUCCESS(olMemUnregister(Device, Arr, Flags));
}

TEST_P(olMemRegisterTest, SuccessMultipleRegisterNoLock) {
  int Arr[50];
  void *PinnedPtr = nullptr;
  ol_memory_register_flags_t Flags = {0};
  ASSERT_SUCCESS(olMemRegister(Device, Arr, sizeof(Arr), Flags, &PinnedPtr));
  ASSERT_SUCCESS(olMemRegister(Device, Arr, sizeof(Arr), Flags, &PinnedPtr));
  ASSERT_SUCCESS(olMemUnregister(Device, Arr, Flags));
  ASSERT_SUCCESS(olMemUnregister(Device, Arr, Flags));
}

TEST_P(olMemRegisterTest, InvalidSizeRegisterNoLock) {
  int Arr[50];
  void *PinnedPtr = nullptr;
  ol_memory_register_flags_t Flags = {0};
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olMemRegister(Device, Arr, 0, Flags, &PinnedPtr));
}

TEST_P(olMemRegisterTest, InvalidPtrRegisterNoLock) {
  int Arr[50];
  void *PinnedPtr = nullptr;
  ol_memory_register_flags_t Flags = {0};
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemRegister(Device, nullptr, sizeof(Arr), Flags, &PinnedPtr));
}

TEST_P(olMemRegisterTest, InvalidPtrUnRegisterNoLock) {
  int Arr[50];
  void *PinnedPtr = nullptr;
  ol_memory_register_flags_t Flags = {0};
  ASSERT_SUCCESS(olMemRegister(Device, Arr, sizeof(Arr), Flags, &PinnedPtr));
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemUnregister(Device, nullptr, Flags));
  ASSERT_SUCCESS(olMemUnregister(Device, Arr, Flags));
}
