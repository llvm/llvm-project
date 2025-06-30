//===------- Offload API tests - olGetKernelInfoSize ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"

using olGetKernelInfoSizeTest = OffloadKernelTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetKernelInfoSizeTest);

TEST_P(olGetKernelInfoSizeTest, SuccessProgram) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetKernelInfoSize(Kernel, OL_KERNEL_INFO_PROGRAM, &Size));
  ASSERT_EQ(Size, sizeof(ol_program_handle_t));
}
