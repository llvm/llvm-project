//===------- Offload API tests - olWaitEvent -====-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olWaitEventTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olWaitEventTest);

TEST_P(olWaitEventTest, Success) {
  uint32_t Src = 42;
  void *DstPtr;

  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(
      olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, sizeof(uint32_t), &DstPtr));
  ASSERT_SUCCESS(
      olMemcpy(Queue, DstPtr, Device, &Src, Host, sizeof(Src), &Event));
  ASSERT_NE(Event, nullptr);
  ASSERT_SUCCESS(olWaitEvent(Event));
  ASSERT_SUCCESS(olDestroyEvent(Event));
}

TEST_P(olWaitEventTest, InvalidNullEvent) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olWaitEvent(nullptr));
}

TEST_P(olWaitEventTest, SuccessMultipleWait) {
  uint32_t Src = 42;
  void *DstPtr;

  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(
      olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, sizeof(uint32_t), &DstPtr));
  ASSERT_SUCCESS(
      olMemcpy(Queue, DstPtr, Device, &Src, Host, sizeof(Src), &Event));
  ASSERT_NE(Event, nullptr);

  for (size_t I = 0; I < 10; I++)
    ASSERT_SUCCESS(olWaitEvent(Event));

  ASSERT_SUCCESS(olDestroyEvent(Event));
}
