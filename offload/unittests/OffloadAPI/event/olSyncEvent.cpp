//===------- Offload API tests - olSyncEvent -====-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olSyncEventTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olSyncEventTest);

TEST_P(olSyncEventTest, Success) {
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, &Event));
  ASSERT_NE(Event, nullptr);
  ASSERT_SUCCESS(olSyncEvent(Event));
  ASSERT_SUCCESS(olDestroyEvent(Event));
}

TEST_P(olSyncEventTest, InvalidNullEvent) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olSyncEvent(nullptr));
}

TEST_P(olSyncEventTest, SuccessMultipleSync) {
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, &Event));
  ASSERT_NE(Event, nullptr);

  for (size_t I = 0; I < 10; I++)
    ASSERT_SUCCESS(olSyncEvent(Event));

  ASSERT_SUCCESS(olDestroyEvent(Event));
}
