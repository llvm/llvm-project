//===------- Offload API tests - olMemcpyWithFlags ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

#include <vector>

using olMemcpyWithFlagsTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olMemcpyWithFlagsTest);

// Copy Size bytes to a fresh device allocation with the given flags, read them
// back with an ordinary memcpy enqueued afterwards, and check the result.
static void checkRoundTrip(ol_context_handle_t Context,
                           ol_device_handle_t Device, ol_device_handle_t Host,
                           ol_queue_handle_t Queue, size_t Size,
                           ol_memcpy_flags_t Flags) {
  void *Alloc;
  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  std::vector<uint8_t> Input(Size);
  for (size_t I = 0; I < Size; ++I)
    Input[I] = static_cast<uint8_t>(I);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(
      olMemcpyWithFlags(Queue, Alloc, Device, Input.data(), Host, Size, Flags));
  ASSERT_SUCCESS(olMemcpy(Queue, Output.data(), Host, Alloc, Device, Size));
  if (Queue)
    ASSERT_SUCCESS(olSyncQueue(Queue));
  ASSERT_EQ(Input, Output);
  ASSERT_SUCCESS(olMemFree(Context, Alloc));
}

TEST_P(olMemcpyWithFlagsTest, SuccessHtoDNoFlags) {
  checkRoundTrip(Context, Device, Host, Queue, 1024, 0);
}

TEST_P(olMemcpyWithFlagsTest, SuccessHtoDDirectWrite) {
  checkRoundTrip(Context, Device, Host, Queue, 1024,
                 OL_MEMCPY_FLAG_DIRECT_WRITE);
}

TEST_P(olMemcpyWithFlagsTest, SuccessHtoDDirectWriteLarge) {
  checkRoundTrip(Context, Device, Host, Queue, 1024 * 1024,
                 OL_MEMCPY_FLAG_DIRECT_WRITE);
}

TEST_P(olMemcpyWithFlagsTest, SuccessHtoDDirectWriteSync) {
  checkRoundTrip(Context, Device, Host, nullptr, 1024,
                 OL_MEMCPY_FLAG_DIRECT_WRITE);
}

TEST_P(olMemcpyWithFlagsTest, SuccessHtoDDirectWriteAfterPendingWork) {
  // Direct-write copies to distinct allocations, behind ordinary work on the
  // queue, must each be observed by the work enqueued after them.
  constexpr size_t Size = 256;
  constexpr int NumCopies = 16;
  void *Allocs[NumCopies];
  std::vector<uint8_t> Inputs[NumCopies];
  std::vector<uint8_t> Outputs[NumCopies];
  void *Pending;
  std::vector<uint8_t> PendingInput(1024 * 1024, 7);
  ASSERT_SUCCESS(olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE,
                            PendingInput.size(), &Pending));
  ASSERT_SUCCESS(olMemcpy(Queue, Pending, Device, PendingInput.data(), Host,
                          PendingInput.size()));

  for (int K = 0; K < NumCopies; ++K) {
    ASSERT_SUCCESS(
        olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, Size, &Allocs[K]));
    Inputs[K].assign(Size, static_cast<uint8_t>(K + 1));
    Outputs[K].assign(Size, 0);
    ASSERT_SUCCESS(olMemcpyWithFlags(Queue, Allocs[K], Device, Inputs[K].data(),
                                     Host, Size, OL_MEMCPY_FLAG_DIRECT_WRITE));
    ASSERT_SUCCESS(
        olMemcpy(Queue, Outputs[K].data(), Host, Allocs[K], Device, Size));
  }
  ASSERT_SUCCESS(olSyncQueue(Queue));

  for (int K = 0; K < NumCopies; ++K) {
    ASSERT_EQ(Inputs[K], Outputs[K]);
    ASSERT_SUCCESS(olMemFree(Context, Allocs[K]));
  }
  ASSERT_SUCCESS(olMemFree(Context, Pending));
}

TEST_P(olMemcpyWithFlagsTest, InvalidFlags) {
  constexpr size_t Size = 64;
  void *Alloc;
  ASSERT_SUCCESS(
      olMemAlloc(Context, Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  std::vector<uint8_t> Input(Size, 42);
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olMemcpyWithFlags(Queue, Alloc, Device, Input.data(), Host, Size,
                                 0xdeadbeef));
  ASSERT_SUCCESS(olMemFree(Context, Alloc));
}
