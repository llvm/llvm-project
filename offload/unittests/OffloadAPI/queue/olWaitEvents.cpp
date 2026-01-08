//===------- Offload API tests - olWaitEvents -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

struct olWaitEventsTest : OffloadProgramTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadProgramTest::SetUpWith("sequence"));
    ASSERT_SUCCESS(
        olGetSymbol(Program, "sequence", OL_SYMBOL_KIND_KERNEL, &Kernel));
    LaunchArgs.Dimensions = 1;
    LaunchArgs.GroupSize = {1, 1, 1};
    LaunchArgs.NumGroups = {1, 1, 1};
    LaunchArgs.DynSharedMemory = 0;
  }

  void TearDown() override {
    RETURN_ON_FATAL_FAILURE(OffloadProgramTest::TearDown());
  }

  ol_symbol_handle_t Kernel = nullptr;
  ol_kernel_launch_size_args_t LaunchArgs{};
};
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olWaitEventsTest);

TEST_P(olWaitEventsTest, Success) {
  constexpr size_t NUM_KERNELS = 16;
  ol_queue_handle_t Queues[NUM_KERNELS];
  ol_event_handle_t Events[NUM_KERNELS];

  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            NUM_KERNELS * sizeof(uint32_t), &Mem));
  struct {
    uint32_t Idx;
    void *Mem;
  } Args{0, Mem};

  for (size_t I = 0; I < NUM_KERNELS; I++) {
    Args.Idx = I;

    ASSERT_SUCCESS(olCreateQueue(Device, &Queues[I]));

    if (I > 0)
      ASSERT_SUCCESS(olWaitEvents(Queues[I], &Events[I - 1], 1));

    ASSERT_SUCCESS(olLaunchKernel(Queues[I], Device, Kernel, &Args,
                                  sizeof(Args), &LaunchArgs));
    ASSERT_SUCCESS(olCreateEvent(Queues[I], &Events[I]));
  }

  ASSERT_SUCCESS(olSyncEvent(Events[NUM_KERNELS - 1]));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 2; i < NUM_KERNELS; i++) {
    ASSERT_EQ(Data[i], Data[i - 1] + Data[i - 2]);
  }
}

TEST_P(olWaitEventsTest, SuccessSingleQueue) {
  constexpr size_t NUM_KERNELS = 16;
  ol_queue_handle_t Queue;
  ol_event_handle_t Events[NUM_KERNELS];

  ASSERT_SUCCESS(olCreateQueue(Device, &Queue));

  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            NUM_KERNELS * sizeof(uint32_t), &Mem));
  struct {
    uint32_t Idx;
    void *Mem;
  } Args{0, Mem};

  for (size_t I = 0; I < NUM_KERNELS; I++) {
    Args.Idx = I;

    if (I > 0)
      ASSERT_SUCCESS(olWaitEvents(Queue, &Events[I - 1], 1));

    ASSERT_SUCCESS(olLaunchKernel(Queue, Device, Kernel, &Args, sizeof(Args),
                                  &LaunchArgs));
    ASSERT_SUCCESS(olCreateEvent(Queue, &Events[I]));
  }

  ASSERT_SUCCESS(olSyncEvent(Events[NUM_KERNELS - 1]));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 2; i < NUM_KERNELS; i++) {
    ASSERT_EQ(Data[i], Data[i - 1] + Data[i - 2]);
  }
}

TEST_P(olWaitEventsTest, SuccessMultipleEvents) {
  constexpr size_t NUM_KERNELS = 16;
  ol_queue_handle_t Queues[NUM_KERNELS];
  ol_event_handle_t Events[NUM_KERNELS];

  void *Mem;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                            NUM_KERNELS * sizeof(uint32_t), &Mem));
  struct {
    uint32_t Idx;
    void *Mem;
  } Args{0, Mem};

  for (size_t I = 0; I < NUM_KERNELS; I++) {
    Args.Idx = I;

    ASSERT_SUCCESS(olCreateQueue(Device, &Queues[I]));

    if (I > 0)
      ASSERT_SUCCESS(olWaitEvents(Queues[I], Events, I));

    ASSERT_SUCCESS(olLaunchKernel(Queues[I], Device, Kernel, &Args,
                                  sizeof(Args), &LaunchArgs));
    ASSERT_SUCCESS(olCreateEvent(Queues[I], &Events[I]));
  }

  ASSERT_SUCCESS(olSyncEvent(Events[NUM_KERNELS - 1]));

  uint32_t *Data = (uint32_t *)Mem;
  for (uint32_t i = 2; i < NUM_KERNELS; i++) {
    ASSERT_EQ(Data[i], Data[i - 1] + Data[i - 2]);
  }
}

TEST_P(olWaitEventsTest, InvalidNullQueue) {
  ol_event_handle_t Event;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olWaitEvents(nullptr, &Event, 1));
}

TEST_P(olWaitEventsTest, InvalidNullEvent) {
  ol_queue_handle_t Queue;
  ASSERT_SUCCESS(olCreateQueue(Device, &Queue));
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER, olWaitEvents(Queue, nullptr, 1));
}

TEST_P(olWaitEventsTest, InvalidNullInnerEvent) {
  ol_queue_handle_t Queue;
  ASSERT_SUCCESS(olCreateQueue(Device, &Queue));
  ol_event_handle_t Event = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olWaitEvents(Queue, &Event, 1));
}
