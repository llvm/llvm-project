//===------- Offload API tests - olReleaseKernel --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olDestroyKernelTest = OffloadKernelTest;

TEST_F(olDestroyKernelTest, Success) {
  ASSERT_SUCCESS(olDestroyKernel(Kernel));
  Kernel = nullptr;
}

TEST_F(olDestroyKernelTest, InvalidNullHandle) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olDestroyKernel(nullptr));
}
