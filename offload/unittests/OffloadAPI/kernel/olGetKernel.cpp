//===------- Offload API tests - olGetKernel ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olGetKernelTest = OffloadProgramTest;

TEST_F(olGetKernelTest, Success) {
  ol_kernel_handle_t Kernel = nullptr;
  ASSERT_SUCCESS(olGetKernel(Program, "foo", &Kernel));
  ASSERT_NE(Kernel, nullptr);
}

TEST_F(olGetKernelTest, InvalidNullProgram) {
  ol_kernel_handle_t Kernel = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetKernel(nullptr, "foo", &Kernel));
}

TEST_F(olGetKernelTest, InvalidNullKernelPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetKernel(Program, "foo", nullptr));
}
