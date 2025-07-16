//===------- Offload API tests - olGetSymbol ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olGetSymbolKernelTest = OffloadProgramTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetSymbolKernelTest);

struct olGetSymbolGlobalTest : OffloadQueueTest {
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
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetSymbolGlobalTest);

TEST_P(olGetSymbolKernelTest, Success) {
  ol_symbol_handle_t Kernel = nullptr;
  ASSERT_SUCCESS(olGetSymbol(Program, "foo", OL_SYMBOL_KIND_KERNEL, &Kernel));
  ASSERT_NE(Kernel, nullptr);
}

TEST_P(olGetSymbolKernelTest, InvalidNullProgram) {
  ol_symbol_handle_t Kernel = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetSymbol(nullptr, "foo", OL_SYMBOL_KIND_KERNEL, &Kernel));
}

TEST_P(olGetSymbolKernelTest, InvalidNullKernelPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetSymbol(Program, "foo", OL_SYMBOL_KIND_KERNEL, nullptr));
}

TEST_P(olGetSymbolKernelTest, InvalidKernelName) {
  ol_symbol_handle_t Kernel = nullptr;
  ASSERT_ERROR(OL_ERRC_NOT_FOUND, olGetSymbol(Program, "invalid_kernel_name",
                                              OL_SYMBOL_KIND_KERNEL, &Kernel));
}

TEST_P(olGetSymbolKernelTest, InvalidKind) {
  ol_symbol_handle_t Kernel = nullptr;
  ASSERT_ERROR(
      OL_ERRC_INVALID_ENUMERATION,
      olGetSymbol(Program, "foo", OL_SYMBOL_KIND_FORCE_UINT32, &Kernel));
}

TEST_P(olGetSymbolGlobalTest, Success) {
  ol_symbol_handle_t Global = nullptr;
  ASSERT_SUCCESS(
      olGetSymbol(Program, "global", OL_SYMBOL_KIND_GLOBAL_VARIABLE, &Global));
  ASSERT_NE(Global, nullptr);
}

TEST_P(olGetSymbolGlobalTest, InvalidNullProgram) {
  ol_symbol_handle_t Global = nullptr;
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_HANDLE,
      olGetSymbol(nullptr, "global", OL_SYMBOL_KIND_GLOBAL_VARIABLE, &Global));
}

TEST_P(olGetSymbolGlobalTest, InvalidNullGlobalPointer) {
  ASSERT_ERROR(
      OL_ERRC_INVALID_NULL_POINTER,
      olGetSymbol(Program, "global", OL_SYMBOL_KIND_GLOBAL_VARIABLE, nullptr));
}

TEST_P(olGetSymbolGlobalTest, InvalidGlobalName) {
  ol_symbol_handle_t Global = nullptr;
  ASSERT_ERROR(OL_ERRC_NOT_FOUND,
               olGetSymbol(Program, "invalid_global",
                           OL_SYMBOL_KIND_GLOBAL_VARIABLE, &Global));
}
