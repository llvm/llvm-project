//===------- Offload API tests - olGetKernelMaxGroupSize ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olKernelGetMaxGroupSizeTest = OffloadKernelTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olKernelGetMaxGroupSizeTest);

TEST_P(olKernelGetMaxGroupSizeTest, Success) {
  size_t Size{0};
  ASSERT_SUCCESS(olGetKernelMaxGroupSize(Device, Kernel, 0, &Size));
  ASSERT_GT(Size, 0u);
}

TEST_P(olKernelGetMaxGroupSizeTest, SuccessMem) {
  size_t Size{0};
  ASSERT_SUCCESS(olGetKernelMaxGroupSize(Device, Kernel, 1024, &Size));
  ASSERT_GT(Size, 0u);
}

TEST_P(olKernelGetMaxGroupSizeTest, NullKernel) {
  size_t Size;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetKernelMaxGroupSize(Device, nullptr, 0, &Size));
}

TEST_P(olKernelGetMaxGroupSizeTest, NullDevice) {
  size_t Size;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetKernelMaxGroupSize(nullptr, Kernel, 0, &Size));
}

TEST_P(olKernelGetMaxGroupSizeTest, NullOutput) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetKernelMaxGroupSize(Device, Kernel, 0, nullptr));
}
