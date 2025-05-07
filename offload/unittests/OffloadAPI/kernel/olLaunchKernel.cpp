//===------- Offload API tests - olLaunchKernel --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

struct olLaunchKernelTest : OffloadQueueTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadQueueTest::SetUp());
    ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Device, DeviceBin));
    ASSERT_GE(DeviceBin->getBufferSize(), 0lu);
    ASSERT_SUCCESS(olCreateProgram(Device, DeviceBin->getBufferStart(),
                                   DeviceBin->getBufferSize(), &Program));
    ASSERT_SUCCESS(olGetKernel(Program, "foo", &Kernel));
    LaunchArgs.Dimensions = 1;
    LaunchArgs.GroupSizeX = 64;
    LaunchArgs.GroupSizeY = 1;
    LaunchArgs.GroupSizeZ = 1;

    LaunchArgs.NumGroupsX = 1;
    LaunchArgs.NumGroupsY = 1;
    LaunchArgs.NumGroupsZ = 1;

    LaunchArgs.DynSharedMemory = 0;
  }

  void TearDown() override {
    if (Program) {
      olDestroyProgram(Program);
    }
    RETURN_ON_FATAL_FAILURE(OffloadQueueTest::TearDown());
  }

  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ol_program_handle_t Program = nullptr;
  ol_kernel_handle_t Kernel = nullptr;
  ol_kernel_launch_size_args_t LaunchArgs{};
};

OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olLaunchKernelTest);

TEST_P(olLaunchKernelTest, Success) {
  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, 64, &Mem));
  struct {
    void *Mem;
  } Args{Mem};

  ASSERT_SUCCESS(olLaunchKernel(Queue, Device, Kernel, &Args, sizeof(Args),
                                &LaunchArgs, nullptr));

  ASSERT_SUCCESS(olWaitQueue(Queue));

  int *Data = (int *)Mem;
  for (int i = 0; i < 64; i++) {
    ASSERT_EQ(Data[i], i);
  }

  ASSERT_SUCCESS(olMemFree(Mem));
}

TEST_P(olLaunchKernelTest, SuccessSynchronous) {
  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED, 64, &Mem));

  struct {
    void *Mem;
  } Args{Mem};

  ASSERT_SUCCESS(olLaunchKernel(nullptr, Device, Kernel, &Args, sizeof(Args),
                                &LaunchArgs, nullptr));

  int *Data = (int *)Mem;
  for (int i = 0; i < 64; i++) {
    ASSERT_EQ(Data[i], i);
  }

  ASSERT_SUCCESS(olMemFree(Mem));
}
