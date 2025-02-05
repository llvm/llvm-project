//===------- Offload API tests - olMemAlloc -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olMemAllocTest = offloadDeviceTest;

TEST_F(olMemAllocTest, SuccessAllocShared) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_SHARED, 1024, &Alloc));
  ASSERT_NE(Alloc, nullptr);
  olMemFree(Device, OL_ALLOC_TYPE_SHARED, Alloc);
}

TEST_F(olMemAllocTest, SuccessAllocHost) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_HOST, 1024, &Alloc));
  ASSERT_NE(Alloc, nullptr);
  olMemFree(Device, OL_ALLOC_TYPE_HOST, Alloc);
}

TEST_F(olMemAllocTest, SuccessAllocDevice) {
  void *Alloc = nullptr;
  ASSERT_SUCCESS(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 1024, &Alloc));
  ASSERT_NE(Alloc, nullptr);
  olMemFree(Device, OL_ALLOC_TYPE_DEVICE, Alloc);
}

TEST_F(olMemAllocTest, InvalidNullDevice) {
  void *Alloc = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_HANDLE,
               olMemAlloc(nullptr, OL_ALLOC_TYPE_DEVICE, 1024, &Alloc));
}

TEST_F(olMemAllocTest, InvalidNullOutPtr) {
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, 1024, nullptr));
}
