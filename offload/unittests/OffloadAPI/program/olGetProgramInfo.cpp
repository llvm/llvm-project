//===------- Offload API tests - olGetProgramInfo -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olGetProgramInfoTest = OffloadProgramTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetProgramInfoTest);

TEST_P(olGetProgramInfoTest, SuccessImageBase) {
  void *Base = nullptr;
  ASSERT_SUCCESS(olGetProgramInfo(Program, OL_PROGRAM_INFO_IMAGE_BASE,
                                  sizeof(Base), &Base));
  ASSERT_NE(Base, nullptr);
}

TEST_P(olGetProgramInfoTest, SuccessImageSize) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetProgramInfo(Program, OL_PROGRAM_INFO_IMAGE_SIZE,
                                  sizeof(Size), &Size));
  ASSERT_GT(Size, 0u);
}

TEST_P(olGetProgramInfoTest, SuccessImageIsReadable) {
  void *Base = nullptr;
  size_t Size = 0;
  ASSERT_SUCCESS(olGetProgramInfo(Program, OL_PROGRAM_INFO_IMAGE_BASE,
                                  sizeof(Base), &Base));
  ASSERT_SUCCESS(olGetProgramInfo(Program, OL_PROGRAM_INFO_IMAGE_SIZE,
                                  sizeof(Size), &Size));
  // The image must stay valid as long as the program; touch every byte.
  const volatile char *Bytes = static_cast<const volatile char *>(Base);
  unsigned Sum = 0;
  for (size_t I = 0; I < Size; ++I)
    Sum += Bytes[I];
  (void)Sum;
}

TEST_P(olGetProgramInfoTest, InvalidNullHandle) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetProgramInfo(nullptr, OL_PROGRAM_INFO_IMAGE_SIZE,
                                sizeof(Size), &Size));
}

TEST_P(olGetProgramInfoTest, InvalidProgramInfoEnumeration) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetProgramInfo(Program, OL_PROGRAM_INFO_FORCE_UINT32,
                                sizeof(Size), &Size));
}

TEST_P(olGetProgramInfoTest, InvalidSizeZero) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetProgramInfo(Program, OL_PROGRAM_INFO_IMAGE_SIZE, 0, &Size));
}

TEST_P(olGetProgramInfoTest, InvalidSizeSmall) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetProgramInfo(Program, OL_PROGRAM_INFO_IMAGE_SIZE,
                                sizeof(Size) - 1, &Size));
}

TEST_P(olGetProgramInfoTest, InvalidNullPointerPropValue) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetProgramInfo(Program, OL_PROGRAM_INFO_IMAGE_SIZE,
                                sizeof(Size), nullptr));
}
