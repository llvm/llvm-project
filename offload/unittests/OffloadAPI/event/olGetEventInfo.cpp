//===------- Offload API tests - olGetEventInfo ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"

using olGetEventInfoTest = OffloadEventTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetEventInfoTest);

TEST_P(olGetEventInfoTest, SuccessQueue) {
  ol_queue_handle_t RetrievedQueue;
  ASSERT_SUCCESS(olGetEventInfo(Event, OL_EVENT_INFO_QUEUE,
                                sizeof(ol_queue_handle_t), &RetrievedQueue));
  ASSERT_EQ(Queue, RetrievedQueue);
}

TEST_P(olGetEventInfoTest, SuccessIsComplete) {
  bool Complete = false;
  while (!Complete) {
    ASSERT_SUCCESS(olGetEventInfo(Event, OL_EVENT_INFO_IS_COMPLETE,
                                  sizeof(Complete), &Complete));
  }
  ASSERT_EQ(Complete, true);
}

TEST_P(olGetEventInfoTest, InvalidNullHandle) {
  ol_queue_handle_t RetrievedQueue;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetEventInfo(nullptr, OL_EVENT_INFO_QUEUE,
                              sizeof(RetrievedQueue), &RetrievedQueue));
}

TEST_P(olGetEventInfoTest, InvalidEventInfoEnumeration) {
  ol_queue_handle_t RetrievedQueue;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetEventInfo(Event, OL_EVENT_INFO_FORCE_UINT32,
                              sizeof(RetrievedQueue), &RetrievedQueue));
}

TEST_P(olGetEventInfoTest, InvalidSizeZero) {
  ol_queue_handle_t RetrievedQueue;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetEventInfo(Event, OL_EVENT_INFO_QUEUE, 0, &RetrievedQueue));
}

TEST_P(olGetEventInfoTest, InvalidSizeSmall) {
  ol_queue_handle_t RetrievedQueue;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetEventInfo(Event, OL_EVENT_INFO_QUEUE,
                              sizeof(RetrievedQueue) - 1, &RetrievedQueue));
}

TEST_P(olGetEventInfoTest, InvalidNullPointerPropValue) {
  ol_queue_handle_t RetrievedQueue;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetEventInfo(Event, OL_EVENT_INFO_QUEUE,
                              sizeof(RetrievedQueue), nullptr));
}
