//===------- Offload API tests - olRetainProgram --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olRetainProgramTest = offloadProgramTest;

TEST_F(olRetainProgramTest, Success) {
  ASSERT_SUCCESS(olRetainProgram(Program));
}

TEST_F(olRetainProgramTest, InvalidNullHandle) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olRetainProgram(nullptr));
}
