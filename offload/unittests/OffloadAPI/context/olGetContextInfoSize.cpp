//===------- Offload API tests - olGetContextInfoSize ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olGetContextInfoSizeTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetContextInfoSizeTest);

TEST_P(olGetContextInfoSizeTest, SuccessNumDevices) {
  size_t Size = 0;
  ASSERT_SUCCESS(
      olGetContextInfoSize(Context, OL_CONTEXT_INFO_NUM_DEVICES, &Size));
  ASSERT_EQ(Size, sizeof(size_t));
}

TEST_P(olGetContextInfoSizeTest, SuccessDevices) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetContextInfoSize(Context, OL_CONTEXT_INFO_DEVICES, &Size));
  ASSERT_EQ(Size, sizeof(ol_device_handle_t));
}

TEST_P(olGetContextInfoSizeTest, SuccessPlatform) {
  size_t Size = 0;
  ASSERT_SUCCESS(
      olGetContextInfoSize(Context, OL_CONTEXT_INFO_PLATFORM, &Size));
  ASSERT_EQ(Size, sizeof(ol_platform_handle_t));
}

TEST_P(olGetContextInfoSizeTest, InvalidNullHandle) {
  size_t Size = 0;
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_HANDLE,
      olGetContextInfoSize(nullptr, OL_CONTEXT_INFO_NUM_DEVICES, &Size));
}

TEST_P(olGetContextInfoSizeTest, InvalidNullOut) {
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_POINTER,
      olGetContextInfoSize(Context, OL_CONTEXT_INFO_NUM_DEVICES, nullptr));
}
