//===------- Offload API tests - olCreateKernel ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olCreateKernelTest = offloadProgramTest;

TEST_F(olCreateKernelTest, Success) {
//   std::shared_ptr<std::vector<char>> DeviceBin2;
//   ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Platform, DeviceBin2));

  ol_kernel_handle_t Kernel = nullptr;
  ASSERT_SUCCESS(olCreateKernel(Program, "foo", &Kernel));
  ASSERT_NE(Kernel, nullptr);
  ASSERT_SUCCESS(olReleaseKernel(Kernel));
}

TEST_F(olCreateKernelTest, InvalidNullProgram) {
  ol_kernel_handle_t Kernel = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olCreateKernel(nullptr, "foo", &Kernel));
}

TEST_F(olCreateKernelTest, InvalidNullKernelPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olCreateKernel(Program, "foo", nullptr));
}

// TEST_F(olCreateKernelTest, InvalidKernelName) {
//   ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Platform, DeviceBin));
//   ol_kernel_handle_t Kernel = nullptr;
//   ASSERT_ANY_ERROR(olCreateKernel(Program, "bad_kernel_name", &Kernel));
//   ASSERT_EQ(Kernel, nullptr);
// }