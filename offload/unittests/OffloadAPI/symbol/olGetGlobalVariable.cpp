//===------- Offload API tests - olGetGlobalVariable ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

struct olGetGlobalVariableTest : OffloadQueueTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadQueueTest::SetUp());
    ASSERT_TRUE(TestEnvironment::loadDeviceBinary("global", Device, DeviceBin));
    ASSERT_GE(DeviceBin->getBufferSize(), 0lu);
    ASSERT_SUCCESS(olCreateProgram(Device, DeviceBin->getBufferStart(),
                                   DeviceBin->getBufferSize(), &Program));
  }

  void TearDown() override {
    if (Program) {
      olDestroyProgram(Program);
    }
    RETURN_ON_FATAL_FAILURE(OffloadQueueTest::TearDown());
  }

  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ol_program_handle_t Program = nullptr;
  ol_kernel_launch_size_args_t LaunchArgs{};
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetGlobalVariableTest);

TEST_P(olGetGlobalVariableTest, Success) {
  ol_symbol_handle_t Global = nullptr;
  ASSERT_SUCCESS(olGetGlobalVariable(Program, "global", &Global));
  ASSERT_NE(Global, nullptr);
}

TEST_P(olGetGlobalVariableTest, InvalidNullProgram) {
  ol_symbol_handle_t Global = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetGlobalVariable(nullptr, "global", &Global));
}

TEST_P(olGetGlobalVariableTest, InvalidNullGlobalPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetGlobalVariable(Program, "global", nullptr));
}

TEST_P(olGetGlobalVariableTest, InvalidGlobalName) {
  ol_symbol_handle_t Global = nullptr;
  ASSERT_ERROR(OL_ERRC_NOT_FOUND,
               olGetGlobalVariable(Program, "invalid_global", &Global));
}
