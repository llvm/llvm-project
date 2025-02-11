//===------- Offload API tests - olEnqueueDataWrite -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olEnqueueDataWriteTest = offloadQueueTest;

TEST_F(olEnqueueDataWriteTest, Success) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  std::vector<uint8_t> Input(Size, 42);
  ASSERT_SUCCESS(olEnqueueDataWrite(Queue, Alloc, Input.data(), Size, nullptr));
  olFinishQueue(Queue);
  olMemFree(Device, OL_ALLOC_TYPE_DEVICE, Alloc);
}
