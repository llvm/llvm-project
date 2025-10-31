//===------- Offload API tests - olCalculateOptimalOccupancy --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olCalculateOptimalOccupancyTest = OffloadKernelTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olCalculateOptimalOccupancyTest);

TEST_P(olCalculateOptimalOccupancyTest, Success) {
  size_t Size{0};
  ASSERT_SUCCESS_OR_UNSUPPORTED(
      olCalculateOptimalOccupancy(Device, Kernel, 0, &Size));
  ASSERT_GT(Size, 0u);
}

TEST_P(olCalculateOptimalOccupancyTest, SuccessMem) {
  size_t Size{0};
  ASSERT_SUCCESS_OR_UNSUPPORTED(
      olCalculateOptimalOccupancy(Device, Kernel, 1024, &Size));
  ASSERT_GT(Size, 0u);
}

TEST_P(olCalculateOptimalOccupancyTest, NullKernel) {
  size_t Size;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olCalculateOptimalOccupancy(Device, nullptr, 0, &Size));
}

TEST_P(olCalculateOptimalOccupancyTest, NullDevice) {
  size_t Size;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olCalculateOptimalOccupancy(nullptr, Kernel, 0, &Size));
}

TEST_P(olCalculateOptimalOccupancyTest, NullOutput) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olCalculateOptimalOccupancy(Device, Kernel, 0, nullptr));
}
