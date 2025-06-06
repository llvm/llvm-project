//===------- Offload API tests - olGetKernelInfo --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"

using olGetKernelInfoTest = OffloadKernelTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetKernelInfoTest);

TEST_P(olGetKernelInfoTest, SuccessProgram) {
  ol_program_handle_t ReadProgram;
  ASSERT_SUCCESS(olGetKernelInfo(Kernel, OL_KERNEL_INFO_PROGRAM,
                                 sizeof(ol_program_handle_t), &ReadProgram));
  ASSERT_EQ(Program, ReadProgram);
}

TEST_P(olGetKernelInfoTest, InvalidNullHandle) {
  ol_program_handle_t Program;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetKernelInfo(nullptr, OL_KERNEL_INFO_PROGRAM, sizeof(Program),
                               &Program));
}

TEST_P(olGetKernelInfoTest, InvalidKernelInfoEnumeration) {
  ol_program_handle_t Program;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetKernelInfo(Kernel, OL_KERNEL_INFO_FORCE_UINT32,
                               sizeof(Program), &Program));
}

TEST_P(olGetKernelInfoTest, InvalidSizeZero) {
  ol_program_handle_t Program;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetKernelInfo(Kernel, OL_KERNEL_INFO_PROGRAM, 0, &Program));
}

TEST_P(olGetKernelInfoTest, InvalidSizeSmall) {
  ol_program_handle_t Program;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetKernelInfo(Kernel, OL_KERNEL_INFO_PROGRAM,
                               sizeof(Program) - 1, &Program));
}

TEST_P(olGetKernelInfoTest, InvalidNullPointerPropValue) {
  ol_program_handle_t Program;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetKernelInfo(Kernel, OL_KERNEL_INFO_PROGRAM, sizeof(Program),
                               nullptr));
}
