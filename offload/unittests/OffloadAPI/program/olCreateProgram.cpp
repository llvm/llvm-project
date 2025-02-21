//===------- Offload API tests - olCreateProgram --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olCreateProgramTest = offloadDeviceTest;

TEST_F(olCreateProgramTest, Success) {

  std::shared_ptr<std::vector<char>> DeviceBin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Platform, DeviceBin));
  ASSERT_GE(DeviceBin->size(), 0lu);

  ol_program_handle_t Program;
  ASSERT_SUCCESS(
      olCreateProgram(Device, DeviceBin->data(), DeviceBin->size(), &Program));
  ASSERT_NE(Program, nullptr);

  ASSERT_SUCCESS(olReleaseProgram(Program));
}
