//===------- Offload API tests - olGetEventElapsedTime --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include "llvm/Support/MemoryBuffer.h"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

namespace {

struct olGetEventElapsedTimeTest : OffloadQueueTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadQueueTest::SetUp());

    ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Device, DeviceBin));
    ASSERT_SUCCESS(olCreateProgram(Device, DeviceBin->getBufferStart(),
                                   DeviceBin->getBufferSize(), &Program));
    ASSERT_SUCCESS(olGetSymbol(Program, "foo", OL_SYMBOL_KIND_KERNEL, &Kernel));

    LaunchArgs.Dimensions = 1;
    LaunchArgs.GroupSize = {64, 1, 1};
    LaunchArgs.NumGroups = {1, 1, 1};
    LaunchArgs.DynSharedMemory = 0;

    ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_MANAGED,
                              LaunchArgs.GroupSize.x * sizeof(uint32_t), &Mem));
  }

  void TearDown() override {
    if (Mem)
      ASSERT_SUCCESS(olMemFree(Mem));
    if (Program)
      ASSERT_SUCCESS(olDestroyProgram(Program));
    RETURN_ON_FATAL_FAILURE(OffloadQueueTest::TearDown());
  }

  void launchFoo() {
    struct {
      void *Mem;
    } Args{Mem};

    ASSERT_SUCCESS(olLaunchKernel(Queue, Device, Kernel, &Args, sizeof(Args),
                                  &LaunchArgs, nullptr));
  }

  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ol_program_handle_t Program = nullptr;
  ol_symbol_handle_t Kernel = nullptr;
  ol_kernel_launch_size_args_t LaunchArgs{};
  void *Mem = nullptr;
};

OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetEventElapsedTimeTest);

TEST_P(olGetEventElapsedTimeTest, Success) {
  ol_event_handle_t StartEvent = nullptr;
  ol_event_handle_t EndEvent = nullptr;

  ASSERT_SUCCESS(olCreateEvent(Queue, &StartEvent));
  ASSERT_NE(StartEvent, nullptr);

  launchFoo();

  ASSERT_SUCCESS(olCreateEvent(Queue, &EndEvent));
  ASSERT_NE(EndEvent, nullptr);

  ASSERT_SUCCESS(olSyncEvent(EndEvent));

  float ElapsedTime = -1.0f;

  ASSERT_SUCCESS(olGetEventElapsedTime(StartEvent, EndEvent, &ElapsedTime));
  ASSERT_GE(ElapsedTime, 0.0f);

  ASSERT_SUCCESS(olDestroyEvent(StartEvent));
  ASSERT_SUCCESS(olDestroyEvent(EndEvent));
}

TEST_P(olGetEventElapsedTimeTest, SuccessMultipleCalls) {
  ol_event_handle_t StartEvent = nullptr;
  ol_event_handle_t EndEvent = nullptr;

  ASSERT_SUCCESS(olCreateEvent(Queue, &StartEvent));
  ASSERT_NE(StartEvent, nullptr);

  launchFoo();

  ASSERT_SUCCESS(olCreateEvent(Queue, &EndEvent));
  ASSERT_NE(EndEvent, nullptr);

  ASSERT_SUCCESS(olSyncEvent(EndEvent));

  float ElapsedTimeA = -1.0f;
  float ElapsedTimeB = -1.0f;

  ASSERT_SUCCESS(olGetEventElapsedTime(StartEvent, EndEvent, &ElapsedTimeA));
  ASSERT_SUCCESS(olGetEventElapsedTime(StartEvent, EndEvent, &ElapsedTimeB));

  ASSERT_GE(ElapsedTimeA, 0.0f);
  ASSERT_GE(ElapsedTimeB, 0.0f);

  ASSERT_SUCCESS(olDestroyEvent(StartEvent));
  ASSERT_SUCCESS(olDestroyEvent(EndEvent));
}

TEST_P(olGetEventElapsedTimeTest, InvalidNullStartEvent) {
  ol_event_handle_t EndEvent = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, &EndEvent));

  float ElapsedTime = 0.0f;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetEventElapsedTime(nullptr, EndEvent, &ElapsedTime));

  ASSERT_SUCCESS(olDestroyEvent(EndEvent));
}

TEST_P(olGetEventElapsedTimeTest, InvalidNullEndEvent) {
  ol_event_handle_t StartEvent = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, &StartEvent));

  float ElapsedTime = 0.0f;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetEventElapsedTime(StartEvent, nullptr, &ElapsedTime));

  ASSERT_SUCCESS(olDestroyEvent(StartEvent));
}

TEST_P(olGetEventElapsedTimeTest, InvalidNullElapsedTime) {
  ol_event_handle_t StartEvent = nullptr;
  ol_event_handle_t EndEvent = nullptr;

  ASSERT_SUCCESS(olCreateEvent(Queue, &StartEvent));
  ASSERT_SUCCESS(olCreateEvent(Queue, &EndEvent));

  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetEventElapsedTime(StartEvent, EndEvent, nullptr));

  ASSERT_SUCCESS(olDestroyEvent(StartEvent));
  ASSERT_SUCCESS(olDestroyEvent(EndEvent));
}

} // namespace
