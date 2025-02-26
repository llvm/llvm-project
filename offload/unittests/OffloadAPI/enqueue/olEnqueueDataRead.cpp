//===------- Offload API tests - olEnqueueMemcpyDtoH ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olEnqueueMemcpyDtoHTest = offloadQueueTest;

TEST_F(olEnqueueMemcpyDtoHTest, Success) {
  constexpr size_t Size = 1024;
  void *Alloc;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  ASSERT_SUCCESS(
      olEnqueueMemcpyHtoD(Queue, Alloc, Input.data(), Size, nullptr));
  ASSERT_SUCCESS(
      olEnqueueMemcpyDtoH(Queue, Output.data(), Alloc, Size, nullptr));
  ASSERT_SUCCESS(olWaitQueue(Queue));
  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
  ASSERT_SUCCESS(olMemFree(Device, OL_ALLOC_TYPE_DEVICE, Alloc));
}
