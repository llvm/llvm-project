//===------- Offload API tests - olMemcpy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olMemcpyTest = OffloadQueueTest;

TEST_F(olMemcpyTest, SuccessHtoD) {
  constexpr size_t Size = 1024;
  void *Alloc;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  std::vector<uint8_t> Input(Size, 42);
  ASSERT_SUCCESS(
      olMemcpy(Queue, Alloc, Device, Input.data(), Host, Size, nullptr));
  olWaitQueue(Queue);
  olMemFree(Alloc);
}

TEST_F(olMemcpyTest, SuccessDtoH) {
  constexpr size_t Size = 1024;
  void *Alloc;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  ASSERT_SUCCESS(
      olMemcpy(Queue, Alloc, Device, Input.data(), Host, Size, nullptr));
  ASSERT_SUCCESS(
      olMemcpy(Queue, Output.data(), Host, Alloc, Device, Size, nullptr));
  ASSERT_SUCCESS(olWaitQueue(Queue));
  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_F(olMemcpyTest, SuccessDtoD) {
  constexpr size_t Size = 1024;
  void *AllocA;
  void *AllocB;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &AllocA));
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &AllocB));
  ASSERT_SUCCESS(
      olMemcpy(Queue, AllocA, Device, Input.data(), Host, Size, nullptr));
  ASSERT_SUCCESS(
      olMemcpy(Queue, AllocB, Device, AllocA, Device, Size, nullptr));
  ASSERT_SUCCESS(
      olMemcpy(Queue, Output.data(), Host, AllocB, Device, Size, nullptr));
  ASSERT_SUCCESS(olWaitQueue(Queue));
  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
  ASSERT_SUCCESS(olMemFree(AllocA));
  ASSERT_SUCCESS(olMemFree(AllocB));
}

TEST_F(olMemcpyTest, SuccessHtoHSync) {
  constexpr size_t Size = 1024;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(olMemcpy(nullptr, Output.data(), Host, Input.data(), Host,
                          Size, nullptr));

  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
}

TEST_F(olMemcpyTest, SuccessDtoHSync) {
  constexpr size_t Size = 1024;
  void *Alloc;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, &Alloc));
  ASSERT_SUCCESS(
      olMemcpy(nullptr, Alloc, Device, Input.data(), Host, Size, nullptr));
  ASSERT_SUCCESS(
      olMemcpy(nullptr, Output.data(), Host, Alloc, Device, Size, nullptr));
  for (uint8_t Val : Output) {
    ASSERT_EQ(Val, 42);
  }
  ASSERT_SUCCESS(olMemFree(Alloc));
}

TEST_F(olMemcpyTest, SuccessSizeZero) {
  constexpr size_t Size = 1024;
  std::vector<uint8_t> Input(Size, 42);
  std::vector<uint8_t> Output(Size, 0);

  // As with std::memcpy, size 0 is allowed. Keep all other arguments valid even
  // if they aren't used.
  ASSERT_SUCCESS(
      olMemcpy(nullptr, Output.data(), Host, Input.data(), Host, 0, nullptr));
}
