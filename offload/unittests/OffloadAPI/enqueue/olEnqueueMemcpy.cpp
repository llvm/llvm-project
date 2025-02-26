//===------- Offload API tests - olEnqueueMemcpy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olEnqueueMemcpyTest = offloadQueueTest;

TEST_F(olEnqueueMemcpyTest, SuccessH2D) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  std::vector<uint8_t> Input(Size, 42);
  ol_device_handle_t Host;
  ASSERT_SUCCESS(olGetHostDevice(&Host));
  ASSERT_SUCCESS(
      olEnqueueMemcpy(Queue, Alloc, Device, Input.data(), Host, Size, nullptr));
  olWaitQueue(Queue);
  olMemFree(Device, OL_ALLOC_TYPE_DEVICE, Alloc);
}

TEST_F(olEnqueueMemcpyTest, SuccessDtoH) {
  constexpr size_t Size = 1024;
  void *Alloc;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);
  ol_device_handle_t Host;
  ASSERT_SUCCESS(olGetHostDevice(&Host));

  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  ASSERT_SUCCESS(
      olEnqueueMemcpy(Queue, Alloc, Device, Input.data(), Host, Size, nullptr));
  ASSERT_SUCCESS(olEnqueueMemcpy(Queue, Output.data(), Host, Alloc, Device,
                                 Size, nullptr));
  ASSERT_SUCCESS(olWaitQueue(Queue));
  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
  ASSERT_SUCCESS(olMemFree(Device, OL_ALLOC_TYPE_DEVICE, Alloc));
}

TEST_F(olEnqueueMemcpyTest, SuccessDtoD) {
  constexpr size_t Size = 1024;
  void *AllocA;
  void *AllocB;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);
  ol_device_handle_t Host;
  ASSERT_SUCCESS(olGetHostDevice(&Host));

  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &AllocA));
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &AllocB));
  ASSERT_SUCCESS(olEnqueueMemcpy(Queue, AllocA, Device, Input.data(), Host,
                                 Size, nullptr));
  ASSERT_SUCCESS(
      olEnqueueMemcpy(Queue, AllocB, Device, AllocA, Device, Size, nullptr));
  ASSERT_SUCCESS(olEnqueueMemcpy(Queue, Output.data(), Host, AllocB, Device,
                                 Size, nullptr));
  ASSERT_SUCCESS(olWaitQueue(Queue));
  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
  ASSERT_SUCCESS(olMemFree(Device, OL_ALLOC_TYPE_DEVICE, AllocA));
  ASSERT_SUCCESS(olMemFree(Device, OL_ALLOC_TYPE_DEVICE, AllocB));
}
