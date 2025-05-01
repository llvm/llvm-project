//===------- Offload API tests - olDestroyEvent ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olDestroyEventTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olDestroyEventTest);

TEST_P(olDestroyEventTest, Success) {
  uint32_t Src = 42;
  void *DstPtr;

  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(
      olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, sizeof(uint32_t), &DstPtr));
  ASSERT_SUCCESS(
      olMemcpy(Queue, DstPtr, Device, &Src, Host, sizeof(Src), &Event));
  ASSERT_NE(Event, nullptr);
  ASSERT_SUCCESS(olWaitQueue(Queue));
  ASSERT_SUCCESS(olDestroyEvent(Event));
}

TEST_P(olDestroyEventTest, InvalidNullEvent) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olDestroyEvent(nullptr));
}
