//===------- Offload API tests - olReleaseProgram -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olReleaseProgramTest = offloadProgramTest;

TEST_F(olReleaseProgramTest, Success) {
  ASSERT_SUCCESS(olRetainProgram(Program));
  ASSERT_SUCCESS(olReleaseProgram(Program));
}

TEST_F(olReleaseProgramTest, InvalidNullHandle) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olReleaseProgram(nullptr));
}
