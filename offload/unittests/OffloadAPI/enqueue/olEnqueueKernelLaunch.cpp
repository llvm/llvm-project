//===------- Offload API tests - olEnqueueKernelLaunch --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

struct olEnqueueKernelLaunchTest : offloadQueueTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(offloadQueueTest::SetUp());
    ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Platform, DeviceBin));
    ASSERT_GE(DeviceBin->size(), 0lu);
    ASSERT_SUCCESS(olCreateProgram(Device, DeviceBin->data(), DeviceBin->size(),
                                   &Program));
    ASSERT_SUCCESS(olCreateKernel(Program, "foo", &Kernel));
  }

  void TearDown() override {
    if (Kernel) {
      olReleaseKernel(Kernel);
    }
    if (Program) {
      olReleaseProgram(Program);
    }
    RETURN_ON_FATAL_FAILURE(offloadQueueTest::TearDown());
  }

  std::shared_ptr<std::vector<char>> DeviceBin;
  ol_program_handle_t Program = nullptr;
  ol_kernel_handle_t Kernel = nullptr;
};

TEST_F(olEnqueueKernelLaunchTest, Success) {
  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_SHARED, 64, &Mem));
  ol_kernel_launch_size_args_t LaunchArgs{};
  LaunchArgs.Dimensions = 1;
  LaunchArgs.GroupSizeX = 64;
  LaunchArgs.GroupSizeY = 1;
  LaunchArgs.GroupSizeZ = 1;

  LaunchArgs.NumGroupsX = 1;
  LaunchArgs.NumGroupsY = 1;
  LaunchArgs.NumGroupsZ = 1;

  struct {
    void *Mem;
  } Args{Mem};

  ASSERT_SUCCESS(olEnqueueKernelLaunch(Queue, Kernel, &Args, sizeof(Args),
                                       &LaunchArgs, nullptr));

  ASSERT_SUCCESS(olWaitQueue(Queue));

  int *Data = (int *)Mem;
  for (int i = 0; i < 64; i++) {
    ASSERT_EQ(Data[i], i);
  }

  ASSERT_SUCCESS(olMemFree(Device, OL_ALLOC_TYPE_SHARED, Mem));
}
