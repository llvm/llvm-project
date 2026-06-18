//===------- Offload API tests - olGetKernel ------------------------------===//
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
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetKernelTest);

using olGetKernelNoProgramTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetKernelNoProgramTest);

TEST_P(olGetKernelTest, Success) {
  ol_kernel_handle_t Kernel = nullptr;
  ASSERT_SUCCESS(olGetKernel(Device, "foo", &Kernel));
  ASSERT_NE(Kernel, nullptr);
}

TEST_P(olGetKernelTest, SuccessSameAsSymbol) {
  ol_kernel_handle_t Kernel = nullptr;
  ol_symbol_handle_t Symbol = nullptr;
  ASSERT_SUCCESS(olGetSymbol(Program, "foo", OL_SYMBOL_KIND_KERNEL, &Symbol));
  ASSERT_SUCCESS(olGetKernel(Device, "foo", &Kernel));
  ASSERT_EQ(Kernel, Symbol);
}

TEST_P(olGetKernelTest, InvalidNullDevice) {
  ol_kernel_handle_t Kernel = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetKernel(nullptr, "foo", &Kernel));
}

TEST_P(olGetKernelTest, InvalidNullName) {
  ol_kernel_handle_t Kernel = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetKernel(Device, nullptr, &Kernel));
}

TEST_P(olGetKernelTest, InvalidNullKernelPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetKernel(Device, "foo", nullptr));
}

TEST_P(olGetKernelTest, InvalidKernelName) {
  ol_kernel_handle_t Kernel = nullptr;
  ASSERT_ERROR(OL_ERRC_NOT_FOUND,
               olGetKernel(Device, "invalid_kernel_name", &Kernel));
}

TEST_P(olGetKernelNoProgramTest, NoLoadedProgram) {
  ol_kernel_handle_t Kernel = nullptr;
  ASSERT_ERROR(OL_ERRC_NOT_FOUND, olGetKernel(Device, "foo", &Kernel));
}
