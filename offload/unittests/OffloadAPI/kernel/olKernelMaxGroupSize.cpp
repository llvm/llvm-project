//===------- Offload API tests - olKernelMaxGroupSize ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olKernelMaxGroupSizeTest = OffloadKernelTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olKernelMaxGroupSizeTest);

TEST_P(olKernelMaxGroupSizeTest, Success) {
  size_t Size{0};
  ASSERT_SUCCESS(olKernelMaxGroupSize(Kernel, Device, 0, &Size));
  ASSERT_GT(Size, 0);
}

TEST_P(olKernelMaxGroupSizeTest, NullKernel) {
  size_t Size;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olKernelMaxGroupSize(nullptr, Device, 0, &Size));
}

TEST_P(olKernelMaxGroupSizeTest, NullDevice) {
  size_t Size;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olKernelMaxGroupSize(Kernel, nullptr, 0, &Size));
}

TEST_P(olKernelMaxGroupSizeTest, NullOutput) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olKernelMaxGroupSize(Kernel, Device, 0, nullptr));
}
