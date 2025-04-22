//===------- Offload API tests - olDestroyProgram -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olDestroyProgramTest = OffloadProgramTest;

TEST_F(olDestroyProgramTest, Success) {
  ASSERT_SUCCESS(olDestroyProgram(Program));
  Program = nullptr;
}

TEST_F(olDestroyProgramTest, InvalidNullHandle) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olDestroyProgram(nullptr));
}
