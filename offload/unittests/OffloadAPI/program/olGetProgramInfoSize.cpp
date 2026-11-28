//===------- Offload API tests - olGetProgramInfoSize ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olGetProgramInfoSizeTest = OffloadProgramTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetProgramInfoSizeTest);

TEST_P(olGetProgramInfoSizeTest, SuccessImageBase) {
  size_t Size = 0;
  ASSERT_SUCCESS(
      olGetProgramInfoSize(Program, OL_PROGRAM_INFO_IMAGE_BASE, &Size));
  ASSERT_EQ(Size, sizeof(void *));
}

TEST_P(olGetProgramInfoSizeTest, SuccessImageSize) {
  size_t Size = 0;
  ASSERT_SUCCESS(
      olGetProgramInfoSize(Program, OL_PROGRAM_INFO_IMAGE_SIZE, &Size));
  ASSERT_EQ(Size, sizeof(size_t));
}

TEST_P(olGetProgramInfoSizeTest, InvalidNullHandle) {
  size_t Size = 0;
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_HANDLE,
      olGetProgramInfoSize(nullptr, OL_PROGRAM_INFO_IMAGE_SIZE, &Size));
}

TEST_P(olGetProgramInfoSizeTest, InvalidProgramInfoEnumeration) {
  size_t Size = 0;
  ASSERT_ERROR(
      OL_ERRC_INVALID_ENUMERATION,
      olGetProgramInfoSize(Program, OL_PROGRAM_INFO_FORCE_UINT32, &Size));
}

TEST_P(olGetProgramInfoSizeTest, InvalidNullPointer) {
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_POINTER,
      olGetProgramInfoSize(Program, OL_PROGRAM_INFO_IMAGE_SIZE, nullptr));
}
