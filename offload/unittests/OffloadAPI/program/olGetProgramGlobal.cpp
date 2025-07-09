//===------- Offload API tests - olGetProgramGlobal -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

struct olGetProgramGlobalTest : OffloadQueueTest {
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
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetProgramGlobalTest);

struct olGetProgramGlobalKernelTest : olGetProgramGlobalTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(olGetProgramGlobalTest::SetUp());

    ASSERT_SUCCESS(olGetKernel(Program, "read", &ReadKernel));
    ASSERT_SUCCESS(olGetKernel(Program, "write", &WriteKernel));

    LaunchArgs.Dimensions = 1;
    LaunchArgs.GroupSize = {64, 1, 1};
    LaunchArgs.NumGroups = {1, 1, 1};

    LaunchArgs.DynSharedMemory = 0;
  }

  ol_kernel_handle_t ReadKernel = nullptr;
  ol_kernel_handle_t WriteKernel = nullptr;
  ol_kernel_launch_size_args_t LaunchArgs{};
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetProgramGlobalKernelTest);

TEST_P(olGetProgramGlobalTest, SuccessGetAddr) {
  void *Addr = nullptr;
  ASSERT_SUCCESS(olGetProgramGlobal(Program, "global", &Addr, nullptr));

  ASSERT_NE(Addr, nullptr);
}

TEST_P(olGetProgramGlobalTest, SuccessGetSize) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetProgramGlobal(Program, "global", nullptr, &Size));

  ASSERT_EQ(Size, 64 * sizeof(uint32_t));
}

TEST_P(olGetProgramGlobalTest, SuccessGetBoth) {
  void *Addr = nullptr;
  size_t Size = 0;
  ASSERT_SUCCESS(olGetProgramGlobal(Program, "global", &Addr, &Size));

  ASSERT_EQ(Size, 64 * sizeof(uint32_t));
  ASSERT_NE(Addr, nullptr);
}

TEST_P(olGetProgramGlobalTest, InvalidNullHandle) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetProgramGlobal(nullptr, "global", nullptr, nullptr));
}

TEST_P(olGetProgramGlobalTest, InvalidNullString) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetProgramGlobal(Program, nullptr, nullptr, nullptr));
}

TEST_P(olGetProgramGlobalTest, InvalidGlobalName) {
  ASSERT_ERROR(OL_ERRC_NOT_FOUND,
               olGetProgramGlobal(Program, "nosuchglobal", nullptr, nullptr));
}

TEST_P(olGetProgramGlobalTest, SuccessRoundTrip) {
  void *Addr = nullptr;
  ASSERT_SUCCESS(olGetProgramGlobal(Program, "global", &Addr, nullptr));

  void *SourceMem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            64 * sizeof(uint32_t), &SourceMem));
  uint32_t *SourceData = (uint32_t *)SourceMem;
  for (auto I = 0; I < 64; I++)
    SourceData[I] = I;

  void *DestMem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            64 * sizeof(uint32_t), &DestMem));

  ASSERT_SUCCESS(olMemcpy(Queue, Addr, Device, SourceMem, Host,
                          64 * sizeof(uint32_t), nullptr));
  ASSERT_SUCCESS(olWaitQueue(Queue));
  ASSERT_SUCCESS(olMemcpy(Queue, DestMem, Host, Addr, Device,
                          64 * sizeof(uint32_t), nullptr));
  ASSERT_SUCCESS(olWaitQueue(Queue));

  uint32_t *DestData = (uint32_t *)DestMem;
  for (uint32_t I = 0; I < 64; I++)
    ASSERT_EQ(DestData[I], I);

  ASSERT_SUCCESS(olMemFree(DestMem));
  ASSERT_SUCCESS(olMemFree(SourceMem));
}

TEST_P(olGetProgramGlobalKernelTest, SuccessWriteGlobal) {
  void *Addr = nullptr;
  ASSERT_SUCCESS(olGetProgramGlobal(Program, "global", &Addr, nullptr));

  void *SourceMem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t),
                            &SourceMem));
  uint32_t *SourceData = (uint32_t *)SourceMem;
  for (auto I = 0; I < 64; I++)
    SourceData[I] = I;

  void *DestMem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t),
                            &DestMem));
  struct {
    void *Mem;
  } Args{DestMem};

  ASSERT_SUCCESS(olMemcpy(Queue, Addr, Device, SourceMem, Host,
                          64 * sizeof(uint32_t), nullptr));
  ASSERT_SUCCESS(olWaitQueue(Queue));
  ASSERT_SUCCESS(olLaunchKernel(Queue, Device, ReadKernel, &Args, sizeof(Args),
                                &LaunchArgs, nullptr));
  ASSERT_SUCCESS(olWaitQueue(Queue));

  uint32_t *DestData = (uint32_t *)DestMem;
  for (uint32_t I = 0; I < 64; I++)
    ASSERT_EQ(DestData[I], I);

  ASSERT_SUCCESS(olMemFree(DestMem));
  ASSERT_SUCCESS(olMemFree(SourceMem));
}

TEST_P(olGetProgramGlobalKernelTest, SuccessReadGlobal) {
  void *Addr = nullptr;
  ASSERT_SUCCESS(olGetProgramGlobal(Program, "global", &Addr, nullptr));

  void *DestMem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t),
                            &DestMem));

  ASSERT_SUCCESS(olLaunchKernel(Queue, Device, WriteKernel, nullptr, 0,
                                &LaunchArgs, nullptr));
  ASSERT_SUCCESS(olWaitQueue(Queue));
  ASSERT_SUCCESS(olMemcpy(Queue, DestMem, Host, Addr, Device,
                          64 * sizeof(uint32_t), nullptr));
  ASSERT_SUCCESS(olWaitQueue(Queue));

  uint32_t *DestData = (uint32_t *)DestMem;
  for (uint32_t I = 0; I < 64; I++)
    ASSERT_EQ(DestData[I], I * 2);

  ASSERT_SUCCESS(olMemFree(DestMem));
}
