//===------- Offload API tests - olCreateEvent ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olCreateEventTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olCreateEventTest);

TEST_P(olCreateEventTest, Success) {
  ol_event_handle_t Event = nullptr;
  ASSERT_SUCCESS(olCreateEvent(Queue, &Event));
  ASSERT_NE(Event, nullptr);
}

TEST_P(olCreateEventTest, InvalidNullQueue) {
  ol_event_handle_t Event;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE, olCreateEvent(nullptr, &Event));
}

TEST_P(olCreateEventTest, InvalidNullDest) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER, olCreateEvent(Queue, nullptr));
}
