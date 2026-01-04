//===------- Offload API tests - olGetEventInfoSize -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"

using olGetEventInfoSizeTest = OffloadEventTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetEventInfoSizeTest);

TEST_P(olGetEventInfoSizeTest, SuccessQueue) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetEventInfoSize(Event, OL_EVENT_INFO_QUEUE, &Size));
  ASSERT_EQ(Size, sizeof(ol_queue_handle_t));
}

TEST_P(olGetEventInfoSizeTest, SuccessIsComplete) {
  size_t Size = 0;
  ASSERT_SUCCESS(olGetEventInfoSize(Event, OL_EVENT_INFO_IS_COMPLETE, &Size));
  ASSERT_EQ(Size, sizeof(bool));
}

TEST_P(olGetEventInfoSizeTest, InvalidNullHandle) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetEventInfoSize(nullptr, OL_EVENT_INFO_QUEUE, &Size));
}

TEST_P(olGetEventInfoSizeTest, InvalidEventInfoEnumeration) {
  size_t Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetEventInfoSize(Event, OL_EVENT_INFO_FORCE_UINT32, &Size));
}

TEST_P(olGetEventInfoSizeTest, InvalidNullPointer) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetEventInfoSize(Event, OL_EVENT_INFO_QUEUE, nullptr));
}
