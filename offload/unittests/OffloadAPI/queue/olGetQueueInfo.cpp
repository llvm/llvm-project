//===------- Offload API tests - olGetQueueInfo ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include "../common/Fixtures.hpp"

using olGetQueueInfoTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetQueueInfoTest);

TEST_P(olGetQueueInfoTest, SuccessDevice) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_SUCCESS(olGetQueueInfo(Queue, OL_QUEUE_INFO_DEVICE,
                                sizeof(ol_device_handle_t), &RetrievedDevice));
  ASSERT_EQ(Device, RetrievedDevice);
}

TEST_P(olGetQueueInfoTest, SuccessEmpty) {
  bool Empty;
  ASSERT_SUCCESS(
      olGetQueueInfo(Queue, OL_QUEUE_INFO_EMPTY, sizeof(Empty), &Empty));
}

TEST_P(olGetQueueInfoTest, InvalidNullHandle) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olGetQueueInfo(nullptr, OL_QUEUE_INFO_DEVICE,
                              sizeof(RetrievedDevice), &RetrievedDevice));
}

TEST_P(olGetQueueInfoTest, InvalidQueueInfoEnumeration) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_ERROR(OL_ERRC_INVALID_ENUMERATION,
               olGetQueueInfo(Queue, OL_QUEUE_INFO_FORCE_UINT32,
                              sizeof(RetrievedDevice), &RetrievedDevice));
}

TEST_P(olGetQueueInfoTest, InvalidSizeZero) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE, olGetQueueInfo(Queue, OL_QUEUE_INFO_DEVICE,
                                                    0, &RetrievedDevice));
}

TEST_P(olGetQueueInfoTest, InvalidSizeSmall) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE,
               olGetQueueInfo(Queue, OL_QUEUE_INFO_DEVICE,
                              sizeof(RetrievedDevice) - 1, &RetrievedDevice));
}

TEST_P(olGetQueueInfoTest, InvalidNullPointerPropValue) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olGetQueueInfo(Queue, OL_QUEUE_INFO_DEVICE,
                              sizeof(RetrievedDevice), nullptr));
}
