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

  // TEMP: This will be split off into an enqueue test
  ol_kernel_handle_t Kernel = nullptr;
  ASSERT_SUCCESS(olCreateKernel(Program, "foo", &Kernel));

  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_SHARED, 1024, &Mem));

  ol_queue_handle_t Queue = nullptr;
  ASSERT_SUCCESS(olCreateQueue(Device, &Queue));
  ol_kernel_launch_size_args_t LaunchArgs{};
  LaunchArgs.Dimensions = 1;
  LaunchArgs.GroupSizeX = 64;
  LaunchArgs.GroupSizeY = 1;
  LaunchArgs.GroupSizeZ = 1;

  LaunchArgs.NumGroupsX = 1;
  LaunchArgs.NumGroupsY = 1;
  LaunchArgs.NumGroupsZ = 1;

  ASSERT_SUCCESS(olSetKernelArgValue(Kernel, 0, sizeof(Mem), &Mem));
  ASSERT_SUCCESS(olEnqueueKernelLaunch(Queue, Kernel, &LaunchArgs, nullptr));

  ASSERT_SUCCESS(olFinishQueue(Queue));

  int *Data = (int *)Mem;
  for (int i = 0; i < 64; i++) {
    ASSERT_EQ(Data[i], i);
  }

  ASSERT_SUCCESS(olReleaseQueue(Queue));
  ASSERT_SUCCESS(olReleaseKernel(Kernel));
  ASSERT_SUCCESS(olReleaseProgram(Program));
  ASSERT_SUCCESS(olMemFree(Device, OL_ALLOC_TYPE_SHARED, Mem));
}
