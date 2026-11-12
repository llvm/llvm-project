//===------- Offload API tests - olGetQueueInfo ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>

#include <algorithm>
#include <map>
#include <vector>

#include "../common/Fixtures.hpp"

using olGetQueueInfoTest = OffloadQueueTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olGetQueueInfoTest);

TEST_P(olGetQueueInfoTest, SuccessDevice) {
  ol_device_handle_t RetrievedDevice;
  ASSERT_SUCCESS(olGetQueueInfo(Queue, OL_QUEUE_INFO_DEVICE,
                                sizeof(ol_device_handle_t), &RetrievedDevice));
  ASSERT_EQ(Device, RetrievedDevice);
}

TEST_P(olGetQueueInfoTest, SuccessContext) {
  ol_context_handle_t RetrievedContext;
  ASSERT_SUCCESS(olGetQueueInfo(Queue, OL_QUEUE_INFO_CONTEXT,
                                sizeof(ol_context_handle_t),
                                &RetrievedContext));
  ASSERT_EQ(Context, RetrievedContext);
}

TEST_P(olGetQueueInfoTest, SuccessEmpty) {
  bool Empty;
  ASSERT_SUCCESS(
      olGetQueueInfo(Queue, OL_QUEUE_INFO_EMPTY, sizeof(Empty), &Empty));
}

TEST_P(olGetQueueInfoTest, SuccessNativeHandlesSpread) {
  // Queues created while all others are still live should be spread evenly
  // over whatever hardware queues the device multiplexes them onto: the
  // busiest and the idlest hardware queue differ by at most one user.
  constexpr size_t NumQueues = 8;
  std::vector<ol_queue_handle_t> Queues(NumQueues, nullptr);
  std::map<void *, size_t> Uses;
  for (auto &Q : Queues) {
    ASSERT_SUCCESS(olCreateQueue(Context, Device, &Q));
    void *Handle = nullptr;
    ol_result_t Result =
        olGetQueueInfo(Q, OL_QUEUE_INFO_NATIVE_HANDLE, sizeof(Handle), &Handle);
    if (Result && Result->Code == OL_ERRC_UNSUPPORTED) {
      for (auto &Other : Queues)
        if (Other)
          ASSERT_SUCCESS(olDestroyQueue(Other));
      GTEST_SKIP() << "Native queue handles are not supported";
    }
    ASSERT_SUCCESS(Result);
    ASSERT_NE(Handle, nullptr);
    ++Uses[Handle];
  }

  auto [Min, Max] = std::minmax_element(
      Uses.begin(), Uses.end(),
      [](const auto &A, const auto &B) { return A.second < B.second; });
  EXPECT_LE(Max->second - Min->second, 1u)
      << Uses.size() << " hardware queues for " << NumQueues << " queues";

  for (auto &Q : Queues)
    ASSERT_SUCCESS(olDestroyQueue(Q));
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

TEST_P(olGetQueueInfoTest, SuccessNativeHandle) {
  void *Handle = nullptr;
  ASSERT_SUCCESS_OR_UNSUPPORTED(olGetQueueInfo(
      Queue, OL_QUEUE_INFO_NATIVE_HANDLE, sizeof(Handle), &Handle));
  ASSERT_NE(Handle, nullptr);
}
