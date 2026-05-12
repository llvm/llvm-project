//===------- Offload API tests - olLaunchHostFunction ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>
#include <thread>

struct olLaunchHostFunctionTest : OffloadQueueTest {};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olLaunchHostFunctionTest);

struct olLaunchHostFunctionKernelTest : OffloadKernelTest {};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olLaunchHostFunctionKernelTest);

TEST_P(olLaunchHostFunctionTest, Success) {
  ASSERT_SUCCESS(olLaunchHostFunction(Queue, [](void *) {}, nullptr));
}

TEST_P(olLaunchHostFunctionTest, SuccessSequence) {
  uint32_t Buff[16] = {1, 1};

  for (auto BuffPtr = &Buff[2]; BuffPtr != &Buff[16]; BuffPtr++) {
    ASSERT_SUCCESS(olLaunchHostFunction(
        Queue,
        [](void *BuffPtr) {
          uint32_t *AsU32 = reinterpret_cast<uint32_t *>(BuffPtr);
          AsU32[0] = AsU32[-1] + AsU32[-2];
        },
        BuffPtr));
  }

  ASSERT_SUCCESS(olSyncQueue(Queue));

  for (uint32_t i = 2; i < 16; i++) {
    ASSERT_EQ(Buff[i], Buff[i - 1] + Buff[i - 2]);
  }
}

TEST_P(olLaunchHostFunctionKernelTest, SuccessBlocking) {
  // Verify that a host kernel can block execution - A host task is created that
  // only resolves when Block is set to false.
  ol_kernel_launch_size_args_t LaunchArgs;
  LaunchArgs.Dimensions = 1;
  LaunchArgs.GroupSize = {64, 1, 1};
  LaunchArgs.NumGroups = {1, 1, 1};
  LaunchArgs.DynSharedMemory = 0;

  ol_queue_handle_t Queue;
  ASSERT_SUCCESS(olCreateQueue(Device, &Queue));

  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            LaunchArgs.GroupSize.x * sizeof(uint32_t), &Mem));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 0; i < 64; i++) {
    Data[i] = 0;
  }

  volatile bool Block = true;
  ASSERT_SUCCESS(olLaunchHostFunction(
      Queue,
      [](void *Ptr) {
        volatile bool *Block =
            reinterpret_cast<volatile bool *>(reinterpret_cast<bool *>(Ptr));

        while (*Block)
          std::this_thread::yield();
      },
      const_cast<bool *>(&Block)));

  struct {
    void *Mem;
  } Args{Mem};
  ASSERT_SUCCESS(
      olLaunchKernel(Queue, Device, Kernel, &Args, sizeof(Args), &LaunchArgs));

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  for (uint32_t i = 0; i < 64; i++) {
    ASSERT_EQ(Data[i], 0);
  }

  Block = false;
  ASSERT_SUCCESS(olSyncQueue(Queue));

  for (uint32_t i = 0; i < 64; i++) {
    ASSERT_EQ(Data[i], i);
  }

  ASSERT_SUCCESS(olDestroyQueue(Queue));
  ASSERT_SUCCESS(olMemFree(Mem));
}

TEST_P(olLaunchHostFunctionTest, InvalidNullCallback) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olLaunchHostFunction(Queue, nullptr, nullptr));
}

TEST_P(olLaunchHostFunctionTest, InvalidNullQueue) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olLaunchHostFunction(nullptr, [](void *) {}, nullptr));
}
