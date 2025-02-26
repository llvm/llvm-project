//===------- Offload API tests - olEnqueueMemcpyDtoD ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olEnqueueMemcpyDtoDTest = offloadQueueTest;

TEST_F(olEnqueueMemcpyDtoDTest, Success) {
  constexpr size_t Size = 1024;
  void *AllocA;
  void *AllocB;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &AllocA));
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &AllocB));
  ASSERT_SUCCESS(
      olEnqueueMemcpyHtoD(Queue, AllocA, Input.data(), Size, nullptr));
  ASSERT_SUCCESS(
      olEnqueueMemcpyDtoD(Queue, Device, AllocB, AllocA, Size, nullptr));
  ASSERT_SUCCESS(
      olEnqueueMemcpyDtoH(Queue, Output.data(), AllocB, Size, nullptr));
  ASSERT_SUCCESS(olWaitQueue(Queue));
  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
  ASSERT_SUCCESS(olMemFree(Device, OL_ALLOC_TYPE_DEVICE, AllocA));
  ASSERT_SUCCESS(olMemFree(Device, OL_ALLOC_TYPE_DEVICE, AllocB));
}
