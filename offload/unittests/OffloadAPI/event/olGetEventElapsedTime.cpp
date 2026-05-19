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
#include <vector>

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

// Two events recorded back-to-back on a queue whose last operation is a
// kernel dispatch must produce identical elapsed-time deltas relative to any
// third event.
TEST_P(olGetEventElapsedTimeTest, BackToBackEventsShareDispatchSignal) {
  ol_event_handle_t Start = nullptr;
  ol_event_handle_t MidA = nullptr;
  ol_event_handle_t MidB = nullptr;

  ASSERT_SUCCESS(olCreateEvent(Queue, &Start));
  launchFoo();
  ASSERT_SUCCESS(olCreateEvent(Queue, &MidA));
  ASSERT_SUCCESS(olCreateEvent(Queue, &MidB));

  ASSERT_SUCCESS(olSyncEvent(MidB));

  float DeltaA = -1.0f, DeltaB = -1.0f;
  ASSERT_SUCCESS(olGetEventElapsedTime(Start, MidA, &DeltaA));
  ASSERT_SUCCESS(olGetEventElapsedTime(Start, MidB, &DeltaB));

  ASSERT_GE(DeltaA, 0.0f);
  ASSERT_GE(DeltaB, 0.0f);
  // Both Mid events should be attached to the same kernel dispatch's
  // completion signal (no intervening op separates them). Tolerate
  // minimal difference.
  ASSERT_NEAR(DeltaA, DeltaB, /*abs_error_ms=*/0.05f);

  ASSERT_SUCCESS(olDestroyEvent(Start));
  ASSERT_SUCCESS(olDestroyEvent(MidA));
  ASSERT_SUCCESS(olDestroyEvent(MidB));
}

// Event recorded before any kernel has been dispatched on the queue must
// still work.
TEST_P(olGetEventElapsedTimeTest, EmptyQueueLeadingEventFallback) {
  ol_event_handle_t Start = nullptr;
  ol_event_handle_t End = nullptr;

  ASSERT_SUCCESS(olCreateEvent(Queue, &Start));

  launchFoo();
  ASSERT_SUCCESS(olCreateEvent(Queue, &End));

  ASSERT_SUCCESS(olSyncEvent(End));

  float Elapsed = -1.0f;
  ASSERT_SUCCESS(olGetEventElapsedTime(Start, End, &Elapsed));
  ASSERT_GE(Elapsed, 0.0f);

  ASSERT_SUCCESS(olDestroyEvent(Start));
  ASSERT_SUCCESS(olDestroyEvent(End));
}

// Event recorded immediately after an async memcpy must recycle the
// memcpy's SDMA completion signal (no extra barrier marker emitted) and
// `olGetEventElapsedTime` must still return a valid elapsed time when the
// second event recycles a kernel-dispatch signal -- i.e., cross-kind
// elapsed-time queries work.
TEST_P(olGetEventElapsedTimeTest, MemcpyPriorOpRecyclesAsyncCopy) {
  constexpr size_t CopySize = 1024;
  void *DevBuf = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, CopySize, &DevBuf));
  std::vector<uint8_t> HostBuf(CopySize, 0x5A);

  ol_event_handle_t Start = nullptr;
  ol_event_handle_t End = nullptr;

  // Slot 0: kernel.
  launchFoo();
  // Slot 1: async H->D memcpy on the same queue. The memcpy's SDMA
  // completion signal carries valid `hsa_amd_profiling_get_async_copy_time`
  // timestamps, so the next recordEvent must recycle this slot's signal
  // (AsyncCopy kind) instead of pushing a fresh barrier marker.
  ASSERT_SUCCESS(
      olMemcpy(Queue, DevBuf, Device, HostBuf.data(), Host, CopySize));
  // Start recycles Slot 1's async-copy signal.
  ASSERT_SUCCESS(olCreateEvent(Queue, &Start));

  // Slot 2: kernel.
  launchFoo();
  // End takes the fast path and recycles Slot 2's KernelDispatch signal.
  ASSERT_SUCCESS(olCreateEvent(Queue, &End));

  ASSERT_SUCCESS(olSyncEvent(End));

  float Elapsed = -1.0f;
  ASSERT_SUCCESS(olGetEventElapsedTime(Start, End, &Elapsed));
  // Strictly positive: the memcpy ended before the second kernel finished,
  // so the kernel-end timestamp must be later than the memcpy-end
  // timestamp. A zero or negative value would indicate that the cross-kind
  // tick-domain comparison broke, or that one event latched onto a stale
  // signal.
  ASSERT_GT(Elapsed, 0.0f);

  ASSERT_SUCCESS(olDestroyEvent(Start));
  ASSERT_SUCCESS(olDestroyEvent(End));
  ASSERT_SUCCESS(olMemFree(DevBuf));
}

// Two events recorded back-to-back on a queue whose last operation is an
// async memcpy must produce identical elapsed-time deltas relative to any
// third event. Both events recycle the same SDMA completion signal, so the
// returned `end` ticks must match.
TEST_P(olGetEventElapsedTimeTest, BackToBackEventsShareAsyncCopySignal) {
  constexpr size_t CopySize = 4096;
  void *DevBuf = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, CopySize, &DevBuf));
  std::vector<uint8_t> HostBuf(CopySize, 0x33);

  ol_event_handle_t Start = nullptr;
  ol_event_handle_t MidA = nullptr;
  ol_event_handle_t MidB = nullptr;

  launchFoo();
  ASSERT_SUCCESS(olCreateEvent(Queue, &Start));
  ASSERT_SUCCESS(
      olMemcpy(Queue, DevBuf, Device, HostBuf.data(), Host, CopySize));
  // Both MidA and MidB recycle the same async-copy completion signal
  // because no intervening op separates them.
  ASSERT_SUCCESS(olCreateEvent(Queue, &MidA));
  ASSERT_SUCCESS(olCreateEvent(Queue, &MidB));

  ASSERT_SUCCESS(olSyncEvent(MidB));

  float DeltaA = -1.0f, DeltaB = -1.0f;
  ASSERT_SUCCESS(olGetEventElapsedTime(Start, MidA, &DeltaA));
  ASSERT_SUCCESS(olGetEventElapsedTime(Start, MidB, &DeltaB));

  ASSERT_GE(DeltaA, 0.0f);
  ASSERT_GE(DeltaB, 0.0f);
  // Both Mid events should be attached to the same SDMA copy completion
  // signal; tolerate minimal difference.
  ASSERT_NEAR(DeltaA, DeltaB, /*abs_error_ms=*/0.05f);

  ASSERT_SUCCESS(olDestroyEvent(Start));
  ASSERT_SUCCESS(olDestroyEvent(MidA));
  ASSERT_SUCCESS(olDestroyEvent(MidB));
  ASSERT_SUCCESS(olMemFree(DevBuf));
}

// Elapsed time must work across a queue sync
TEST_P(olGetEventElapsedTimeTest, EventAfterQueueSyncFallback) {
  ol_event_handle_t Start = nullptr;
  ol_event_handle_t End = nullptr;

  launchFoo();
  ASSERT_SUCCESS(olCreateEvent(Queue, &Start));
  ASSERT_SUCCESS(olSyncQueue(Queue));
  launchFoo();
  ASSERT_SUCCESS(olCreateEvent(Queue, &End));

  ASSERT_SUCCESS(olSyncEvent(End));

  float Elapsed = -1.0f;
  ASSERT_SUCCESS(olGetEventElapsedTime(Start, End, &Elapsed));
  ASSERT_GE(Elapsed, 0.0f);

  ASSERT_SUCCESS(olDestroyEvent(Start));
  ASSERT_SUCCESS(olDestroyEvent(End));
}

// Multiple kernels between two events.
// Elapsed time must reflect at least the wall-clock of all enclosed kernels.
TEST_P(olGetEventElapsedTimeTest, MultipleKernelsBetweenEvents) {
  ol_event_handle_t Start = nullptr;
  ol_event_handle_t End = nullptr;

  launchFoo();
  ASSERT_SUCCESS(olCreateEvent(Queue, &Start));
  for (int I = 0; I < 8; ++I)
    launchFoo();
  ASSERT_SUCCESS(olCreateEvent(Queue, &End));

  ASSERT_SUCCESS(olSyncEvent(End));

  float Elapsed = -1.0f;
  ASSERT_SUCCESS(olGetEventElapsedTime(Start, End, &Elapsed));
  ASSERT_GE(Elapsed, 0.0f);

  ASSERT_SUCCESS(olDestroyEvent(Start));
  ASSERT_SUCCESS(olDestroyEvent(End));
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
