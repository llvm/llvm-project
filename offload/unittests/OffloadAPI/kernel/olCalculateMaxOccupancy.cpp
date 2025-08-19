//===------- Offload API tests - olCalculateMaxOccupancy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olCalculateMaxOccupancyTest = OffloadKernelTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olCalculateMaxOccupancyTest);

TEST_P(olCalculateMaxOccupancyTest, Success) {
  size_t Size{0};
  ASSERT_SUCCESS_OR_UNSUPPORTED(olCalculateMaxOccupancy(Device, Kernel, 0, &Size));
  ASSERT_GT(Size, 0u);
}

TEST_P(olCalculateMaxOccupancyTest, SuccessMem) {
  size_t Size{0};
  ASSERT_SUCCESS_OR_UNSUPPORTED(olCalculateMaxOccupancy(Device, Kernel, 1024, &Size));
  ASSERT_GT(Size, 0u);
}

TEST_P(olCalculateMaxOccupancyTest, NullKernel) {
  size_t Size;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olCalculateMaxOccupancy(Device, nullptr, 0, &Size));
}

TEST_P(olCalculateMaxOccupancyTest, NullDevice) {
  size_t Size;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olCalculateMaxOccupancy(nullptr, Kernel, 0, &Size));
}

TEST_P(olCalculateMaxOccupancyTest, NullOutput) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olCalculateMaxOccupancy(Device, Kernel, 0, nullptr));
}
