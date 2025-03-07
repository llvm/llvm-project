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

using olCreateKernelTest = OffloadProgramTest;

TEST_F(olCreateKernelTest, Success) {
  ol_kernel_handle_t Kernel = nullptr;
  ASSERT_SUCCESS(olCreateKernel(Program, "foo", &Kernel));
  ASSERT_NE(Kernel, nullptr);
  ASSERT_SUCCESS(olDestroyKernel(Kernel));
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
